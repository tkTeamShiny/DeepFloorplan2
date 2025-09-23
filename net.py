============================
[Train] images discovered: 29
[Train] usable pairs: 26
  [sample#1] img=/content/DeepFloorplan2/YOLO_SetA_seg_no1_部屋種別改変版_flat/train/images/1000001.jpg
               room=/content/DeepFloorplan2/YOLO_SetA_seg_no1_部屋種別改変版_flat/train/labels/1000001_multi.png
               cw=None
  [sample#2] img=/content/DeepFloorplan2/YOLO_SetA_seg_no1_部屋種別改変版_flat/train/images/1000002.jpg
               room=/content/DeepFloorplan2/YOLO_SetA_seg_no1_部屋種別改変版_flat/train/labels/1000002_multi.png
               cw=None
  [sample#3] img=/content/DeepFloorplan2/YOLO_SetA_seg_no1_部屋種別改変版_flat/train/images/1000003.jpg
               room=/content/DeepFloorplan2/YOLO_SetA_seg_no1_部屋種別改変版_flat/train/labels/1000003_multi.png
               cw=None
  [sample#4] img=/content/DeepFloorplan2/YOLO_SetA_seg_no1_部屋種別改変版_flat/train/images/1000004.jpg
               room=/content/DeepFloorplan2/YOLO_SetA_seg_no1_部屋種別改変版_flat/train/labels/1000004_multi.png
               cw=None
  [sample#5] img=/content/DeepFloorplan2/YOLO_SetA_seg_no1_部屋種別改変版_flat/train/images/1000008.jpg
               room=/content/DeepFloorplan2/YOLO_SetA_seg_no1_部屋種別改変版_flat/train/labels/1000008_multi.png
               cw=None
2025-09-23 06:24:20.992834: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:47] Overriding orig_value setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
I0000 00:00:1758608660.994167    5714 gpu_device.cc:2019] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 79261 MB memory:  -> device: 0, name: NVIDIA A100-SXM4-80GB, pci bus id: 0000:00:05.0, compute capability: 8.0
Traceback (most recent call last):
  File "/content/DeepFloorplan2/main.py", line 486, in <module>
    main()
  File "/content/DeepFloorplan2/main.py", line 481, in main
    train(cfg)
  File "/content/DeepFloorplan2/main.py", line 380, in train
    losses = _train_step(model, optimizer, images, labels_cw, labels_r)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/tmp/__autograph_generated_file118i9wzq.py", line 12, in tf___train_step
    logits_cw, logits_r = ag__.converted_call(ag__.ld(model), (ag__.ld(images),), dict(training=True), fscope)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler
    raise e.with_traceback(filtered_tb) from None
        ^^^^^^^^^^^
  File "/content/DeepFloorplan2/net.py", line 135, in call
    s1 = self.enc1(x0, 64)
         ^^^^^^^^^^^^^^^^^
  File "/content/DeepFloorplan2/net.py", line 28, in _res_block
    h = _conv_bn_relu(x, filters, k, 1, d)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/DeepFloorplan2/net.py", line 23, in _conv_bn_relu
    x = L.Conv2D(filters, k, strides=s, padding="same", dilation_rate=d, use_bias=False)(x)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: in user code:

    File "/content/DeepFloorplan2/main.py", line 342, in _train_step  *
        logits_cw, logits_r = model(images, training=True)
    File "/usr/local/lib/python3.12/dist-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler  **
        raise e.with_traceback(filtered_tb) from None
    File "/content/DeepFloorplan2/net.py", line 135, in call
        s1 = self.enc1(x0, 64)
    File "/content/DeepFloorplan2/net.py", line 28, in _res_block
        h = _conv_bn_relu(x, filters, k, 1, d)
    File "/content/DeepFloorplan2/net.py", line 23, in _conv_bn_relu
        x = L.Conv2D(filters, k, strides=s, padding="same", dilation_rate=d, use_bias=False)(x)

    ValueError: Exception encountered when calling Network.call().
    
    tf.function only supports singleton tf.Variables created on the first call. Make sure the tf.Variable is only created once or created outside tf.function. See https://www.tensorflow.org/guide/function#creating_tfvariables for more information.
    
    Arguments received by Network.call():
      • x=tf.Tensor(shape=(4, 512, 512, 3), dtype=float32)
      • training=True
