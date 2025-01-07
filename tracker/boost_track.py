"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import os
from copy import deepcopy
from typing import Optional, List
import logging
import numpy as np

import cv2
from default_settings import GeneralSettings, BoostTrackSettings, BoostTrackPlusPlusSettings
from tracker.embedding import EmbeddingComputer
from tracker.assoc import associate, iou_batch, MhDist_similarity, shape_similarity, soft_biou_batch
from tracker.ecc import ECC
from tracker.kalmanfilter import KalmanFilter


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,h,r] where x,y is the centre of the box and h is the height and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0

    r = w / float(h + 1e-6)

    return np.array([x, y, h, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,h,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """

    h = x[2]
    r = x[3]
    w = 0 if r <= 0 else r * h

    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0
    active_ids = set()  # 현재 활성화된 ID들을 추적

    @classmethod
    def reset_count(cls):
        """Reset the tracker counter and active IDs."""
        cls.count = 0
        cls.active_ids.clear()

    @classmethod
    def get_next_id(cls):
        """Get next available ID."""
        while cls.count in cls.active_ids:
            cls.count += 1
        cls.active_ids.add(cls.count)
        return cls.count

    def __init__(self, bbox, emb: Optional[np.ndarray] = None):
        """
        Initialises a tracker using initial bounding box.
        """
        self.bbox_to_z_func = convert_bbox_to_z
        self.x_to_bbox_func = convert_x_to_bbox

        self.time_since_update = 0
        self.id = self.get_next_id()
        print(f"Created new tracker with ID: {self.id}")  # Debug output

        self.kf = KalmanFilter(self.bbox_to_z_func(bbox))
        self.emb = emb
        self.hit_streak = 0
        self.age = 0

    def __del__(self):
        """Cleanup when tracker is deleted."""
        KalmanBoxTracker.active_ids.discard(self.id)

    def get_confidence(self, coef: float = 0.9) -> float:
        n = 7

        if self.age < n:
            return coef ** (n - self.age)
        return coef ** (self.time_since_update-1)

    def update(self, bbox: np.ndarray, score: float = 0):
        """
        Updates the state vector with observed bbox.
        """

        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(self.bbox_to_z_func(bbox), score)

    def camera_update(self, transform: np.ndarray):
        x1, y1, x2, y2 = self.get_state()[0]
        x1_, y1_, _ = transform @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = transform @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.kf.x[:4] = [cx, cy, h,  w / h]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_state()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def update_emb(self, emb, alpha=0.9):
        if self.emb is None:
            self.emb = emb
        else:
            self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb


class BoostTrack(object):
    def __init__(self, video_name: Optional[str] = None, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize a tracker.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.inactive_trackers = []  # 비활성 트래커를 별도로 관리
        self.frame_count = 0
        KalmanBoxTracker.reset_count()

        # Reset tracker counter when initializing new BoostTrack instance
        KalmanBoxTracker.reset_count()

        self.det_thresh = GeneralSettings['det_thresh']

        self.lambda_iou = BoostTrackSettings['lambda_iou']
        self.lambda_mhd = BoostTrackSettings['lambda_mhd']
        self.lambda_shape = BoostTrackSettings['lambda_shape']
        self.use_dlo_boost = BoostTrackSettings['use_dlo_boost']
        self.use_duo_boost = BoostTrackSettings['use_duo_boost']
        self.dlo_boost_coef = BoostTrackSettings['dlo_boost_coef']

        self.use_rich_s = BoostTrackPlusPlusSettings['use_rich_s']
        self.use_sb = BoostTrackPlusPlusSettings['use_sb']
        self.use_vt = BoostTrackPlusPlusSettings['use_vt']

        if GeneralSettings['use_embedding']:
            self.embedder = EmbeddingComputer(GeneralSettings['dataset'], GeneralSettings['test_dataset'], True)
        else:
            self.embedder = None

        if GeneralSettings['use_ecc']:
            self.ecc = ECC(scale=350, video_name=video_name, use_cache=True)
        else:
            self.ecc = None
            
            
        self.reid_model = GeneralSettings['reid_model']

        # Initialize logging
        self.logger = logging.getLogger('BoostTrack')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    def update(self, dets, img_tensor, img_numpy, tag):
        self.frame_count += 1
        
        # 현재 활성화된 트래커들의 예측 위치 계산
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        # 비활성 트래커 관리
        current_inactive = []
        for trk in self.trackers:
            if trk.time_since_update > 1 and trk.time_since_update <= self.max_age * 2:
                if trk not in self.inactive_trackers:  # 중복 방지
                    current_inactive.append(trk)
        self.inactive_trackers = current_inactive
        
        # 모든 트래커 목록 생성 (활성 + 비활성)
        all_trackers = self.trackers + self.inactive_trackers
        all_trks = np.zeros((len(all_trackers), 5))
        for t, trk in enumerate(all_trackers):
            pos = trk.get_state()[0]
            all_trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            
        print(f"\n현재 트래커 ID: {[t.id for t in self.trackers]}")
        print(f"비활성 트래커 ID: {[t.id for t in self.inactive_trackers]}")
        
        if self.use_dlo_boost:
            dets = self.dlo_confidence_boost(dets, self.use_rich_s, self.use_sb, self.use_vt)
        
        if self.use_duo_boost:
            dets = self.duo_confidence_boost(dets)
        
        remain_inds = dets[:, 4] >= self.det_thresh
        dets = dets[remain_inds]
        scores = dets[:, 4]
        
        # Generate embeddings for all trackers
        dets_embs = np.ones((dets.shape[0], 1))
        emb_cost = None
        if self.embedder and dets.size > 0:
            dets_embs = self.embedder.compute_embedding(img_numpy, dets[:, :4], tag)
            all_trk_embs = []
            for t in range(len(all_trackers)):
                all_trk_embs.append(all_trackers[t].get_emb())
            all_trk_embs = np.array(all_trk_embs)
            if all_trk_embs.size > 0 and dets.size > 0:
                emb_cost = dets_embs.reshape(dets_embs.shape[0], -1) @ all_trk_embs.reshape((all_trk_embs.shape[0], -1)).T
        
        # Calculate confidence for all trackers
        all_confs = np.array([t.get_confidence() for t in all_trackers])
        
        # Get matching matrices for all trackers
        iou_matrix = iou_batch(dets, all_trks[:, :4])
        mh_matrix = self.get_mh_dist_matrix(dets, all_trackers)
        
        # Perform matching with all trackers
        matched, unmatched_dets, unmatched_trks, cost_matrix = associate(
            dets,
            all_trks,
            self.iou_threshold,
            mahalanobis_distance=mh_matrix,
            track_confidence=all_confs,
            detection_confidence=scores,
            emb_cost=emb_cost,
            lambda_iou=self.lambda_iou,
            lambda_mhd=self.lambda_mhd,
            lambda_shape=self.lambda_shape
        )
        
        # Debug output
        print(f"\n{'='*50}")
        print(f"프레임 {self.frame_count}")
        print(f"현재 트래커 ID: {[t.id for t in self.trackers]}")
        print(f"비활성 트래커 ID: {[t.id for t in self.inactive_trackers]}\n")
        
        if len(matched) > 0:
            print("[매칭된 정보]")
            print(f"검출-트래커 쌍: {matched}")
            print(f"매칭된 트래커 ID: {[all_trackers[t].id for t in matched[:, 1]]}\n")
        
        if len(unmatched_dets) > 0:
            print(f"[매칭되지 않은 검출]: {unmatched_dets}\n")
        
        if len(unmatched_trks) > 0:
            print("[매칭되지 않은 트래커]")
            print(f"인덱스: {unmatched_trks}")
            print(f"트래커 ID: {[all_trackers[t].id for t in unmatched_trks]}\n")
        
        print("=== 상세 매칭 정보 ===\n")
        print("[임베딩 유사도 행렬]")
        print(emb_cost if emb_cost is not None else "임베딩 정보 없음")
        print("\n[IOU 행렬]")
        print(iou_matrix)
        print("\n[마할라노비스 거리 행렬]")
        print(mh_matrix)
        
        print("\n[최종 매칭 결과]")
        for m in matched:
            det_idx, trk_idx = m
            trk_id = all_trackers[trk_idx].id
            print(f"검출 {det_idx} -> 트래커 ID {trk_id} (임베딩: {emb_cost[det_idx, trk_idx]:.4f}, "
                  f"IOU: {iou_matrix[det_idx, trk_idx]:.4f}, "
                  f"마할라노비스: {mh_matrix[det_idx, trk_idx]:.4f})")
        
        print(f"\n{'='*50}")
        
        # Update matched trackers with assigned detections
        for m in matched:
            det_idx, trk_idx = m
            all_trackers[trk_idx].update(dets[det_idx])
            if dets_embs is not None:
                all_trackers[trk_idx].update_emb(dets_embs[det_idx])
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            if dets_embs is not None:
                trk.update_emb(dets_embs[i])
            self.trackers.append(trk)
        
        # Update tracker states and remove dead trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i-1)
            i -= 1
        
        # Generate output
        ret = []
        for trk in self.trackers:
            if trk.time_since_update < 1:
                d = trk.get_state()[0]
                conf = trk.get_confidence()
                ret.append(np.concatenate((d, [trk.id, conf])).reshape(1, -1))
        
        print(f"전체 트래커 ID (잠재적인 것 포함): {[t.id for t in self.trackers]}")
        print(f"시각화되는 트래커 ID: {[int(r[0, -2]) for r in ret if len(r) > 0]}")
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))  # x1, y1, x2, y2, id, confidence

    def dump_cache(self):
        if self.ecc is not None:
            self.ecc.save_cache()

    def get_iou_matrix(self, detections: np.ndarray, buffered: bool = False) -> np.ndarray:
        trackers = np.zeros((len(self.trackers), 5))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], self.trackers[t].get_confidence()]

        return iou_batch(detections, trackers) if not buffered else soft_biou_batch(detections, trackers)

    def get_mh_dist_matrix(self, detections: np.ndarray, trackers: List[KalmanBoxTracker]) -> np.ndarray:
        if len(trackers) == 0:
            return np.zeros((0, 0))
        z = np.zeros((len(detections), 4), dtype=float)
        x = np.zeros((len(trackers), 4), dtype=float)
        sigma_inv = np.zeros_like(x, dtype=float)

        f = trackers[0].bbox_to_z_func
        for i in range(len(detections)):
            z[i, :] = f(detections[i, :]).reshape((-1, ))[:4]
        for i in range(len(trackers)):
            x[i] = trackers[i].kf.x[:4]
            # Note: we assume diagonal covariance matrix
            sigma_inv[i] = np.reciprocal(np.diag(trackers[i].kf.covariance[:4, :4]))

        return ((z.reshape((-1, 1, 4)) - x.reshape((1, -1, 4))) ** 2 * sigma_inv.reshape((1, -1, 4))).sum(axis=2)

    def duo_confidence_boost(self, detections: np.ndarray) -> np.ndarray:
        n_dims = 4
        limit = 13.2767
        mahalanobis_distance = self.get_mh_dist_matrix(detections, self.trackers)

        if mahalanobis_distance.size > 0 and self.frame_count > 1:
            min_mh_dists = mahalanobis_distance.min(1)

            mask = (min_mh_dists > limit) & (detections[:, 4] < self.det_thresh)
            boost_detections = detections[mask]
            boost_detections_args = np.argwhere(mask).reshape((-1,))
            iou_limit = 0.3
            if len(boost_detections) > 0:
                bdiou = iou_batch(boost_detections, boost_detections) - np.eye(len(boost_detections))
                bdiou_max = bdiou.max(axis=1)

                remaining_boxes = boost_detections_args[bdiou_max <= iou_limit]
                args = np.argwhere(bdiou_max > iou_limit).reshape((-1,))
                for i in range(len(args)):
                    boxi = args[i]
                    tmp = np.argwhere(bdiou[boxi] > iou_limit).reshape((-1,))
                    args_tmp = np.append(np.intersect1d(boost_detections_args[args], boost_detections_args[tmp]), boost_detections_args[boxi])

                    conf_max = np.max(detections[args_tmp, 4])
                    if detections[boost_detections_args[boxi], 4] == conf_max:
                        remaining_boxes = np.array(remaining_boxes.tolist() + [boost_detections_args[boxi]])

                mask = np.zeros_like(detections[:, 4], dtype=np.bool_)
                mask[remaining_boxes] = True

            detections[:, 4] = np.where(mask, self.det_thresh + 1e-4, detections[:, 4])

        return detections

    def dlo_confidence_boost(self, detections: np.ndarray, use_rich_sim: bool, use_soft_boost: bool, use_varying_th: bool) -> np.ndarray:
        sbiou_matrix = self.get_iou_matrix(detections, True)
        if sbiou_matrix.size == 0:
            return detections
        trackers = np.zeros((len(self.trackers), 6))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, self.trackers[t].time_since_update - 1]

        if use_rich_sim:
            mhd_sim = MhDist_similarity(self.get_mh_dist_matrix(detections, self.trackers), 1)
            shape_sim = shape_similarity(detections, trackers)
            S = (mhd_sim + shape_sim + sbiou_matrix) / 3
        else:
            S = self.get_iou_matrix(detections, False)

        if not use_soft_boost and not use_varying_th:
            max_s = S.max(1)
            coef = self.dlo_boost_coef
            detections[:, 4] = np.maximum(detections[:, 4], max_s * coef)

        else:
            if use_soft_boost:
                max_s = S.max(1)
                alpha = 0.65
                detections[:, 4] = np.maximum(detections[:, 4], alpha*detections[:, 4] + (1-alpha)*max_s**(1.5))
            if use_varying_th:
                threshold_s = 0.95
                threshold_e = 0.8
                n_steps = 20
                alpha = (threshold_s - threshold_e) / n_steps
                tmp = (S > np.maximum(threshold_s - trackers[:, 5] * alpha, threshold_e)).max(1)
                scores = deepcopy(detections[:, 4])
                scores[tmp] = np.maximum(scores[tmp], self.det_thresh + 1e-5)

                detections[:, 4] = scores

        return detections
