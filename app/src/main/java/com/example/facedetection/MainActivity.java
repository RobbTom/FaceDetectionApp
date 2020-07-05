package com.example.facedetection;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.*;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;

    boolean startFaces = false;
    boolean firstTimeFaces = false;

    Net detector;

    // Upload file to storage and return a path.
    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream;
        try {
            // Read data from assets
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();

            // Create copy file in storage
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();

            // Return a path
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i("TAG", "Error during uploading file");
        }
        return "";
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraBridgeViewBase = (JavaCameraView) findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);

                if (status == BaseLoaderCallback.SUCCESS) {
                    cameraBridgeViewBase.enableView();
                } else {
                    super.onManagerConnected(status);
                }
            }
        };
    }


    public void findFaces(View Button) {
        if (!startFaces) {
            startFaces = true;

            if (!firstTimeFaces) {
                firstTimeFaces = true;
                String protoFile = getPath("deploy.prototxt",this);
                String caffeWeights = getPath("res10_300x300_ssd_iter_140000.caffemodel",this);
                detector = Dnn.readNetFromCaffe(protoFile, caffeWeights);
            }

        } else {
            startFaces = false;
        }
    }


    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // Get a new frame
        Mat frame = inputFrame.rgba();

        if (startFaces) {
            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

            // Forward image through network
            Mat imageBlob = Dnn.blobFromImage(frame,1.0, new Size(300,300),
                    new Scalar(104.0, 177.0, 123.0), true, false, CvType.CV_32F);

            //Set the input through the network
            detector.setInput(imageBlob);
            Mat detections = detector.forward();

            //Getting values of images's dimension
            int cols = frame.cols();
            int rows = frame.rows();

            double THRESHOLD = 0.5;

            detections = detections.reshape(1, (int) detections.total() / 7);

            for (int i = 0; i < detections.rows(); ++i) {
                double confidence = detections.get(i, 2)[0];
                if (confidence > THRESHOLD) {

                    int left = (int) (detections.get(i, 3)[0] * cols);
                    int top = (int) (detections.get(i, 4)[0] * rows);
                    int right = (int) (detections.get(i, 5)[0] * cols);
                    int bottom = (int) (detections.get(i, 6)[0] * rows);

                    // Draw rectangle around face
                    Imgproc.rectangle(frame, new Point(left, top),
                            new Point(right, bottom), new Scalar(255, 255, 0), 2);

                    //Round confidence value
                    BigDecimal bd = new BigDecimal(Double.toString(confidence));
                    bd = bd.setScale(2, RoundingMode.HALF_UP);

                    String label = bd.doubleValue() * 100 + "%";
                    int[] baseLine = new int[1];

                    Size labelSize = Imgproc.getTextSize(label,
                            Core.FONT_HERSHEY_SIMPLEX, 1.5, 1, baseLine);

                    // Draw background for label
                    Imgproc.rectangle(frame, new Point(left, top - labelSize.height),
                                    new Point(left + labelSize.width, top + baseLine[0]),
                                    new Scalar(255, 255, 255), -1 );

                    // Write confidence
                    Imgproc.putText(frame, label, new Point(left, top),
                            Core.FONT_HERSHEY_SIMPLEX, 1.5, new Scalar(0, 0, 0));
                }
            }
        }
        return frame;
    }


    @Override
    public void onCameraViewStarted(int width, int height) {
    }


    @Override
    public void onCameraViewStopped() {
    }


    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(getApplicationContext(), "Error", Toast.LENGTH_SHORT).show();
        } else {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
    }
}