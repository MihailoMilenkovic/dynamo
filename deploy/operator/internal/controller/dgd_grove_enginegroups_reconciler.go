/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

// groveEngineGroupsReconciler owns child creation and binding, never live scale targets or child status.
type groveEngineGroupsReconciler struct {
	client client.Client
}

// Reconcile creates each world once and binds it after Grove publishes its member clique.
// dgd and pcs must not be nil. A missing asynchronous dependency is pending, not a failed world.
func (r *groveEngineGroupsReconciler) Reconcile(ctx context.Context, dgd *api.DynamoGraphDeployment, pcs *grove.PodCliqueSet) (map[string]api.ComponentReplicaStatus, bool, error) {
	statuses := make(map[string]api.ComponentReplicaStatus)
	ready := true
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if component.EngineGroup == nil {
			continue
		}

		// Resolve declarative geometry before creating a durable child or binding native resources.
		profile, err := dynamo.ResolveComponentEngineGroupProfile(component)
		if err != nil {
			return nil, false, fmt.Errorf("component %q: %w", component.ComponentName, err)
		}
		name := dynamo.EngineGroupNameForComponent(dgd.Name, component.ComponentName)
		group := &api.DynamoGraphDeploymentEngineGroup{}
		key := client.ObjectKey{Namespace: dgd.Namespace, Name: name}
		if err := r.client.Get(ctx, key, group); err != nil {
			if !apierrors.IsNotFound(err) {
				return nil, false, fmt.Errorf("get Engine Group %s: %w", key, err)
			}

			// Only creation seeds replicas and policy; subsequent DGD reconciles preserve both.
			group = &api.DynamoGraphDeploymentEngineGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name: name, Namespace: dgd.Namespace,
					Labels: map[string]string{
						consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
						consts.KubeLabelDynamoComponent:           component.ComponentName,
						consts.KubeLabelDynamoEngineGroupRuntime:  consts.KubeLabelDynamoEngineGroupSGLang,
					},
					Annotations: map[string]string{consts.KubeAnnotationDynamoEngineGroupProfile: profile.Geometry.Fingerprint},
					Finalizers:  []string{engineGroupFinalizer},
				},
				Spec: api.DynamoGraphDeploymentEngineGroupSpec{Replicas: component.EngineGroup.InitialSize},
			}
			if policy := component.EngineGroup.Policy; policy != nil {
				group.Spec.Policy = &api.EngineGroupScalingPolicy{MinReplicas: ptr.To(ptr.Deref(policy.MinSize, profile.InitialReplicas)), MaxReplicas: ptr.To(ptr.Deref(policy.MaxSize, profile.MaximumReplicas))}
			}
			// Verify through the graph's stable frontend Service, not an ephemeral frontend Pod IP.
			for j := range dgd.Spec.Components {
				frontend := &dgd.Spec.Components[j]
				if string(frontend.ComponentType) == consts.ComponentTypeFrontend {
					group.Annotations[consts.KubeAnnotationDynamoEngineGroupVerifyURL] = fmt.Sprintf(
						"http://%s.%s.svc:8000/v1/completions", dynamo.GetDCDResourceName(dgd, frontend.ComponentName, ""), dgd.Namespace,
					)
					break
				}
			}
			if err := controllerutil.SetControllerReference(dgd, group, r.client.Scheme()); err != nil {
				return nil, false, fmt.Errorf("set Engine Group owner: %w", err)
			}
			if err := r.client.Create(ctx, group); err != nil {
				if !apierrors.IsAlreadyExists(err) {
					return nil, false, fmt.Errorf("create Engine Group %s: %w", key, err)
				}
				statuses[component.ComponentName] = api.ComponentReplicaStatus{ReadyReplicas: ptr.To(int32(0))}
			} else {
				statuses[component.ComponentName] = api.ComponentReplicaStatus{Replicas: 1, ReadyReplicas: ptr.To(int32(0))}
			}
			ready = false
			continue
		}

		// Never adopt another controller's world or silently reinterpret an existing launch profile.
		if !metav1.IsControlledBy(group, dgd) || group.Annotations[consts.KubeAnnotationDynamoEngineGroupProfile] != profile.Geometry.Fingerprint {
			return nil, false, fmt.Errorf("Engine Group %s has conflicting ownership or immutable profile", key)
		}
		if !group.DeletionTimestamp.IsZero() {
			return nil, false, fmt.Errorf("Engine Group %s is terminating; automatic world recreation is unsupported", key)
		}
		bound, err := r.bindClique(ctx, group, pcs)
		if err != nil {
			return nil, false, err
		}

		// Available, fresh, engine-authoritative state makes a recovering world Ready without requiring its full target.
		available := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable)
		known := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionTopologyKnown)
		worldReady := bound && available != nil && known != nil &&
			available.Status == metav1.ConditionTrue && known.Status == metav1.ConditionTrue &&
			available.ObservedGeneration == group.Generation && known.ObservedGeneration == group.Generation
		status := api.ComponentReplicaStatus{Replicas: 1, ReadyReplicas: ptr.To(int32(0))}
		if group.Status.Profile != nil {
			allocatedGPUs := int64(group.Status.Replicas) * group.Status.Profile.GPUsPerReplica
			status.GPUsPerReplica = ptr.To(allocatedGPUs)
			status.GPUsPerEngine = ptr.To(allocatedGPUs)
		}
		if worldReady {
			status.ReadyReplicas = ptr.To(int32(1))
		}
		statuses[component.ComponentName] = status
		ready = ready && worldReady
	}
	return statuses, ready, nil
}

func (r *groveEngineGroupsReconciler) bindClique(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup, pcs *grove.PodCliqueSet) (bool, error) {
	// Select by the rendered world label, not a guessed Pod name or a mutable rank ordinal.
	cliques := &grove.PodCliqueList{}
	if err := r.client.List(ctx, cliques, client.InNamespace(group.Namespace), client.MatchingLabels{consts.KubeLabelDynamoEngineGroup: group.Name}); err != nil {
		return false, fmt.Errorf("list Engine Group member cliques: %w", err)
	}
	if len(cliques.Items) == 0 {
		return false, nil
	}
	if len(cliques.Items) != 1 {
		return false, fmt.Errorf("Engine Group %s requires exactly one member clique", group.Name)
	}
	clique := &cliques.Items[0]
	owner := metav1.GetControllerOf(clique)
	if owner == nil || owner.Kind != "PodCliqueScalingGroup" || owner.APIVersion != grove.SchemeGroupVersion.String() || clique.UID == "" {
		return false, fmt.Errorf("member clique %s has invalid world ownership or UID", clique.Name)
	}

	// Verify the complete native ownership chain before accepting a capacity binding.
	world := &grove.PodCliqueScalingGroup{}
	if err := r.client.Get(ctx, client.ObjectKey{Namespace: group.Namespace, Name: owner.Name}, world); err != nil {
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		return false, fmt.Errorf("get member-clique world: %w", err)
	}
	if world.UID != owner.UID || !metav1.IsControlledBy(world, pcs) {
		return false, fmt.Errorf("member clique %s is not owned by the DGD's Grove world", clique.Name)
	}
	name := group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodClique]
	uid := group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID]
	if name != "" || uid != "" {
		if name != clique.Name || uid != string(clique.UID) {
			return false, fmt.Errorf("Engine Group %s member-clique incarnation changed; recovery is required", group.Name)
		}
		return true, nil
	}

	// Persist the exact binding once; no later reconcile can rebind a recreated clique.
	before := group.DeepCopy()
	group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodClique] = clique.Name
	group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID] = string(clique.UID)
	if err := r.client.Patch(ctx, group, client.MergeFromWithOptions(before, client.MergeFromWithOptimisticLock{})); err != nil {
		return false, fmt.Errorf("bind Engine Group member clique: %w", err)
	}
	return true, nil
}
