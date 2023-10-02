"""
license: MIT
author: Gabriele Beltramo

Overview
--------
Make instances of torchdata.dataloader2.DataLoader2

"""

import torchdata.dataloader2 as dataloader2


def make(
    datapipe,
    num_workers: int = 1,
    worker_prefetch_cnt: int = 0,
    main_prefetch_cnt: int = 0,
) -> dataloader2.DataLoader2:
    """Make a DataLoader2 using a DataPipe."""

    reading_service = dataloader2.MultiProcessingReadingService(
        num_workers=num_workers,
        worker_prefetch_cnt=worker_prefetch_cnt,
        main_prefetch_cnt=main_prefetch_cnt,
    )

    loader = dataloader2.DataLoader2(datapipe=datapipe, reading_service=reading_service)

    return loader
