"""WIDER+WFLW 통합 학습용 혼합 배치 샘플러 (RetinaFace 원본 data/combined.py).

두 데이터셋을 `ConcatDataset([wider, wflw])` 로 합치면 전역 인덱스
`[0, n_wider)` 가 WIDER, `[n_wider, n_wider+n_wflw)` 가 WFLW 를 가리킨다.
`MixedBatchSampler` 는 배치 내 WIDER:WFLW 구성을 `mix_ratio` 로 고정한다(예: 4.0
→ 4:1). WIDER 는 epoch 당 1회 커버하고, 더 작은 WFLW 풀은 재활용한다.
"""

import math
import random

from torch.utils.data import Sampler


class MixedBatchSampler(Sampler):
    def __init__(self, n_wider, n_wflw, batch_size, mix_ratio=4.0, seed=42):
        if n_wider <= 0 or n_wflw <= 0:
            raise ValueError("MixedBatchSampler needs non-empty WIDER and WFLW sets")
        self.n_wider = n_wider
        self.n_wflw = n_wflw
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = 0  # __iter__ 마다 증가시켜 epoch 별로 다르게 셔플

        # per-batch split: WIDER : WFLW = mix_ratio : 1
        self.b_wflw = max(1, round(batch_size / (mix_ratio + 1.0)))
        self.b_wider = max(1, batch_size - self.b_wflw)
        self._num_batches = math.ceil(n_wider / self.b_wider)

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        wider = list(range(self.n_wider))
        wflw = list(range(self.n_wider, self.n_wider + self.n_wflw))
        rng.shuffle(wider)
        rng.shuffle(wflw)

        wf_ptr = 0
        for i in range(self._num_batches):
            batch = wider[i * self.b_wider:(i + 1) * self.b_wider]
            for _ in range(self.b_wflw):
                if wf_ptr >= len(wflw):  # recycle the minority pool
                    rng.shuffle(wflw)
                    wf_ptr = 0
                batch.append(wflw[wf_ptr])
                wf_ptr += 1
            rng.shuffle(batch)
            yield batch
