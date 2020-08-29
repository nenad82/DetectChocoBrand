/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.choco.android.ml;


import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import io.fotoapparat.Fotoapparat;
import io.fotoapparat.FotoapparatSwitcher;
import io.fotoapparat.facedetector.view.RectanglesView;
import io.fotoapparat.parameter.LensPosition;
import io.fotoapparat.preview.Frame;
import io.fotoapparat.preview.FrameProcessor;
import io.fotoapparat.view.CameraView;

import static io.fotoapparat.log.Loggers.fileLogger;
import static io.fotoapparat.log.Loggers.logcat;
import static io.fotoapparat.log.Loggers.loggers;
import static io.fotoapparat.parameter.selector.LensPositionSelectors.lensPosition;

import com.choco.android.ml.env.Logger;
import com.choco.android.ml.tflite.Classifier.Device;
import com.choco.android.ml.tflite.Classifier.Model;

import com.choco.android.ml.tflite.Classifier;

import com.choco.android.ml.tflite.GpuDelegateHelper;

public class CameraActivity extends AppCompatActivity {

  private static final int INPUT_SIZE = 224;

  private Classifier classifier;
  private Executor executor = Executors.newSingleThreadExecutor();
  private TextView textViewResult;
  private Button btnDetectObject, btnToggleCamera;
  private ImageView imageViewResult;
  private RectanglesView rectanglesView;

  private final PermissionsDelegate permissionsDelegate = new PermissionsDelegate(this);
  private boolean hasCameraPermission;
  private CameraView cameraView;
  private FotoapparatSwitcher fotoapparatSwitcher;
  private Fotoapparat frontFotoapparat;
  private Fotoapparat backFotoapparat;

  private static final Logger LOGGER = new Logger();
  private Handler handler;
  private HandlerThread handlerThread;
  private Bitmap croppedBitmap = null;

  private void recreateClassifier(Model model, Device device, int numThreads) {
    if (classifier != null) {
      LOGGER.d("Closing classifier.");
      classifier.close();
      classifier = null;
    }
    if (device == Device.GPU) {
      if (!GpuDelegateHelper.isGpuDelegateAvailable()) {
        LOGGER.d("Not creating classifier: GPU support unavailable.");
        runOnUiThread(
                () -> {
                  Toast.makeText(this, "GPU acceleration unavailable.", Toast.LENGTH_LONG).show();
                });
        return;
      } else if (model == Model.QUANTIZED && device == Device.GPU) {
        LOGGER.d("Not creating classifier: GPU doesn't support quantized models.");
        runOnUiThread(
                () -> {
                  Toast.makeText(
                          this, "GPU does not yet supported quantized models.", Toast.LENGTH_LONG)
                          .show();
                });
        return;
      }
    }
    try {
      LOGGER.d(
              "Creating classifier (model=%s, device=%s, numThreads=%d)", model, device, numThreads);
      classifier = Classifier.create(this, model, device, numThreads);
    } catch (IOException e) {
      LOGGER.e(e, "Failed to create classifier.");
    }
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_camera);
    cameraView = findViewById(R.id.camera_view);
    rectanglesView =  findViewById(R.id.rectanglesView);
    hasCameraPermission = permissionsDelegate.hasCameraPermission();

    if (hasCameraPermission) {
      cameraView.setVisibility(View.VISIBLE);
    } else {
      permissionsDelegate.requestCameraPermission();
    }

    frontFotoapparat = createFotoapparat(LensPosition.FRONT);
    backFotoapparat = createFotoapparat(LensPosition.BACK);
    fotoapparatSwitcher = FotoapparatSwitcher.withDefault(backFotoapparat);

    View switchCameraButton = findViewById(R.id.switchCamera);
    switchCameraButton.setVisibility(
            canSwitchCameras()
                    ? View.VISIBLE
                    : View.GONE
    );

    switchCameraButton.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        switchCamera();
      }
    });

    textViewResult = findViewById(R.id.recognize_result);

    initTensorFlowAndLoadModel();
/*
        cameraView.addCameraKitListener(new CameraKitEventListener() {
            @Override
            public void onEvent(CameraKitEvent cameraKitEvent) {

            }

            @Override
            public void onError(CameraKitError cameraKitError) {

            }

            @Override
            public void onImage(CameraKitImage cameraKitImage) {

                Bitmap bitmap = cameraKitImage.getBitmap();

                bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);

                imageViewResult.setImageBitmap(bitmap);

                final List<Classifier.Recognition> results = classifier.recognizeImage(bitmap);

                textViewResult.setText(results.toString());

            }


            @Override
            public void onVideo(CameraKitVideo cameraKitVideo) {
                Log.d("testing-tag", String.valueOf(cameraKitVideo.getVideoFile()));
            }
        });

        btnToggleCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraView.toggleFacing();
            }
        });

        btnDetectObject.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraView.captureImage();
            }
        });*/
  }

  private boolean canSwitchCameras() {
    return frontFotoapparat.isAvailable() == backFotoapparat.isAvailable();
  }

  private Fotoapparat createFotoapparat(LensPosition position) {
    return Fotoapparat
            .with(this)
            .into(cameraView)
            .lensPosition(lensPosition(position))
            .frameProcessor(new SampleFrameProcessor()
            )
            .logger(loggers(
                    logcat(),
                    fileLogger(this)
            ))

            .build();

  }

  private class SampleFrameProcessor implements FrameProcessor {
    @Override
    public void processFrame(@NonNull Frame frame) {
      YuvImage yuvImage = new YuvImage(frame.image, ImageFormat.NV21, frame.size.width, frame.size.height, null);
      ByteArrayOutputStream os = new ByteArrayOutputStream();
      yuvImage.compressToJpeg(new Rect(0, 0, frame.size.width, frame.size.height), 100, os);
      byte[] jpegByteArray = os.toByteArray();
      Bitmap bitmap = BitmapFactory.decodeByteArray(jpegByteArray, 0, jpegByteArray.length);
      croppedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);

      //imageViewResult.setImageBitmap(bitmap);

      runInBackground(
        new Runnable() {
          @Override
          public void run() {
            if (classifier != null) {
              final List<Classifier.Recognition> results = classifier.recognizeImage(croppedBitmap);
              runOnUiThread(new Runnable() {
                @Override
                public void run() {

                  if(results.size() > 0) {
                    if(results.get(0).getConfidence() > 0.5)
                      textViewResult.setText(results.get(0).toString());
                    else
                      textViewResult.setText("");
                  } else {
                    textViewResult.setText("");
                  }
                }
              });
            }
            }
          });
    }
  }

  private void switchCamera() {
    if (fotoapparatSwitcher.getCurrentFotoapparat() == frontFotoapparat) {
      fotoapparatSwitcher.switchTo(backFotoapparat);
    } else {
      fotoapparatSwitcher.switchTo(frontFotoapparat);
    }
  }

  @Override
  protected void onResume() {
    super.onResume();
    if (hasCameraPermission) {
      fotoapparatSwitcher.start();
    }

    handlerThread = new HandlerThread("inference");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
  }

  @Override
  protected void onPause() {

    super.onPause();
    if (hasCameraPermission) {
      fotoapparatSwitcher.stop();
    }

    handlerThread.quitSafely();
    try {
      handlerThread.join();
      handlerThread = null;
      handler = null;
    } catch (final InterruptedException e) {
      LOGGER.e(e, "Exception!");
    }
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
    executor.execute(new Runnable() {
      @Override
      public void run() {
        classifier.close();
      }
    });
  }

  @Override
  public void onRequestPermissionsResult(int requestCode,
                                         @NonNull String[] permissions,
                                         @NonNull int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (permissionsDelegate.resultGranted(requestCode, permissions, grantResults)) {
      fotoapparatSwitcher.start();
      cameraView.setVisibility(View.VISIBLE);
    }
  }

  protected synchronized void runInBackground(final Runnable r) {
    if (handler != null) {
      handler.post(r);
    }
  }

  private void initTensorFlowAndLoadModel() {

    recreateClassifier(Model.FLOAT, Device.CPU, 2);

  }
}

