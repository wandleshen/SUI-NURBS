import numpy as np

ctrlpts = np.array(
    [
        [
            [-3.5, -2.5, 1.3988070487976074, 0.3539999723434448],
            [-3.5, -1.5, 1.3988070487976074, 0.3539999723434448],
            [-3.5, -0.5, 1.3988070487976074, 0.7070000171661377],
            [-3.5, 0.5, 1.3988070487976074, 0.7070000171661377],
            [-3.5, 1.5, 1.3988070487976074, 0.3539999723434448],
            [-3.5, 2.5, 1.3988070487976074, 0.3539999723434448],
        ],
        [
            [-2.5, -2.5, 1.3988070487976074, 0.3539999723434448],
            [-2.5, -1.5, 1.3988070487976074, 0.3539999723434448],
            [-2.5, -0.5, 1.3988070487976074, 0.7070000171661377],
            [-2.5, 0.5, 1.3988070487976074, 0.7070000171661377],
            [-2.5, 1.5, 1.3988070487976074, 0.3539999723434448],
            [-2.5, 2.5, 1.3988070487976074, 0.3539999723434448],
        ],
        [
            [-1.5, -2.5, -0.7697777152061462, 0.3539999723434448],
            [-1.5, -1.5, -0.7697777152061462, 0.3539999723434448],
            [-1.5, -0.5, 0.7234739065170288, 0.7070000171661377],
            [-1.5, 0.5, 0.7234739065170288, 0.7070000171661377],
            [-1.5, 1.5, -0.7697777152061462, 0.3539999723434448],
            [-1.5, 2.5, -0.7697777152061462, 0.3539999723434448],
        ],
        [
            [-0.5, -2.5, -0.7697777152061462, 0.7070000171661377],
            [-0.5, -1.5, -0.7697777152061462, 0.7070000171661377],
            [-0.5, -0.5, 0.23022225499153137, 1.4140000343322754],
            [-0.5, 0.5, 0.23022225499153137, 1.4140000343322754],
            [-0.5, 1.5, -0.7697777152061462, 0.7070000171661377],
            [-0.5, 2.5, -0.7697777152061462, 0.7070000171661377],
        ],
        [
            [0.5, -2.5, -0.7697777152061462, 0.7070000171661377],
            [0.5, -1.5, -0.7697777152061462, 0.7070000171661377],
            [0.5, -0.5, -1.2580353021621704, 1.4140000343322754],
            [0.5, 0.5, -1.2580353021621704, 1.4140000343322754],
            [0.5, 1.5, -0.7697777152061462, 0.7070000171661377],
            [0.5, 2.5, -0.7697777152061462, 0.7070000171661377],
        ],
        [
            [1.5, -2.5, -0.7697777152061462, 0.3539999723434448],
            [1.5, -1.5, -0.7697777152061462, 0.3539999723434448],
            [1.5, -0.5, -0.7697777152061462, 0.7070000171661377],
            [1.5, 0.5, -0.7697777152061462, 0.7070000171661377],
            [1.5, 1.5, -0.7697777152061462, 0.3539999723434448],
            [1.5, 2.5, -0.7697777152061462, 0.3539999723434448],
        ],
        [
            [2.5, -2.5, -1.7270164489746094, 0.3539999723434448],
            [2.5, -1.5, -1.7270164489746094, 0.3539999723434448],
            [2.5, -0.5, -1.7270164489746094, 0.7070000171661377],
            [2.5, 0.5, -1.7270164489746094, 0.7070000171661377],
            [2.5, 1.5, -1.7270164489746094, 0.3539999723434448],
            [2.5, 2.5, -1.7270164489746094, 0.3539999723434448],
        ],
    ]
)

ctrlpts[..., 0:3] *= ctrlpts[..., 3, None]

np.save("data/test.npy", ctrlpts)
