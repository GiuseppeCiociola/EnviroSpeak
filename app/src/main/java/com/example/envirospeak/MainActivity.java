package com.example.envirospeak;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.Executor;

import org.opencv.android.OpenCVLoader;


public class MainActivity extends AppCompatActivity implements View.OnClickListener, ImageAnalysis.Analyzer {

    private PreviewView pview;
    private boolean analysis_on;
    private ListenableFuture<ProcessCameraProvider> provider;
    private Yolov5Detector yolov5Detector;
    private MiDaSDepthEstimator depthEstimator;
    Paint boxPaint = new Paint();
    Paint textPain = new Paint();
    private static final int PERMISSION_REQUEST_CODE = 200;
    private ArrayList<Recognition> recognitionsInOrder;
    private TextToSpeech textToSpeech;

    static{
        if(!OpenCVLoader.initDebug())
            Log.d("Error", "Unable to load openCV");
        else
            Log.d("SUCCESS", "openCV loaded");
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button startCamera = findViewById(R.id.startCamera);
        startCamera.setOnClickListener(this);

        Button analyze = findViewById(R.id.det);
        analyze.setOnClickListener(this);

        Button btnSpeak = findViewById(R.id.speak);
        btnSpeak.setOnClickListener(this);

        pview = findViewById(R.id.previewView);
        this.analysis_on = false;

        yolov5Detector = new Yolov5Detector();
        yolov5Detector.setModelFile("yolov5s-fp16.tflite");
        yolov5Detector.initialModel(this);

        boxPaint.setStrokeWidth(5);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setColor(Color.RED);

        textPain.setTextSize(50);
        textPain.setColor(Color.GREEN);
        textPain.setStyle(Paint.Style.FILL);

        try {
            depthEstimator = new MiDaSDepthEstimator(this, "midas.tflite");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        textToSpeech = new TextToSpeech(getApplicationContext(), status -> {
            if (status != TextToSpeech.ERROR) {
                textToSpeech.setLanguage(Locale.ENGLISH);
            }
        });
    }
    @Override
    public void onClick(View v) {
        if(v.getId() == R.id.startCamera){
            if(!checkPermission()) {
                requestPermission();
            }
            provider = ProcessCameraProvider.getInstance(this);
            provider.addListener( () ->
            {
                try{
                    ProcessCameraProvider cameraProvider = provider.get();
                    startCamera(cameraProvider);
                } catch (Exception e)
                {
                    e.printStackTrace();
                }
            }, getExecutor());
        }else if(v.getId() == R.id.det){
            this.analysis_on = !this.analysis_on;
        }else{
            this.analysis_on = !this.analysis_on;
            String introPhrase = "The object detected, starting from the closest one, are as follows:";
            textToSpeech.speak(introPhrase, TextToSpeech.QUEUE_FLUSH, null, null);
            for (Recognition recognition : recognitionsInOrder) {
                textToSpeech.speak(recognition.getLabelName(), TextToSpeech.QUEUE_ADD, null, null);
            }
        }
    }
    private boolean checkPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }
    private void requestPermission() {
        ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.CAMERA},
                PERMISSION_REQUEST_CODE);
    }
    private void startCamera(ProcessCameraProvider cameraProvider) {
        cameraProvider.unbindAll();
        CameraSelector camSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build();
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(pview.getSurfaceProvider());
        ImageCapture imageCapt = new ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build();
        ImageAnalysis imageAn = new ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build();
        imageAn.setAnalyzer(getExecutor(), this);
        cameraProvider.bindToLifecycle(this, camSelector, preview, imageCapt, imageAn);
    }
    private Executor getExecutor() {
        return ContextCompat.getMainExecutor(this);
    }
    private float[][] resizeDepthMap(float[][] depthMap, int targetHeight, int targetWidth) {
        int depthMapHeight = depthMap.length;
        int depthMapWidth = depthMap[0].length;

        float[][] resizedDepthMap = new float[targetHeight][targetWidth];

        float x_ratio = (float) (depthMapWidth - 1) / (targetWidth - 1);
        float y_ratio = (float) (depthMapHeight - 1) / (targetHeight - 1);

        for (int y = 0; y < targetHeight - 1; y++) {
            for (int x = 0; x < targetWidth - 1; x++) {
                //estrazione coordinate dei quattro punti
                int x_l = (int) Math.floor(x_ratio * x);
                int x_h = (int) Math.ceil(x_ratio * x);
                int y_l = (int) Math.floor(y_ratio * y);
                int y_h = (int) Math.ceil(y_ratio * y);
                //estrazione dei valori
                float A = depthMap[y_l][x_l];
                float B = depthMap[y_l][x_h];
                float C = depthMap[y_h][x_l];
                float D = depthMap[y_h][x_h];
                //calcolo dei pesi
                float xWeight = (x_ratio * x) - x_l;
                float yWeight = (y_ratio * y) - y_l;
                //interpolazione bilineare
                resizedDepthMap[y][x] = bilinearInterpolation(xWeight, yWeight, A, B, C, D);
            }
        }
        return resizedDepthMap;
    }
    private float bilinearInterpolation(float xWeight, float yWeight, float A, float B, float C, float D) {
        float topInterpolation = (1 - xWeight) * A + xWeight * B;
        float bottomInterpolation = (1 - xWeight) * C + xWeight * D;
        return (1 - yWeight) * topInterpolation + yWeight * bottomInterpolation;
    }
    @Override
    public void analyze(@NonNull ImageProxy image) {
        Bitmap conv = pview.getBitmap();
        if(this.analysis_on)
        {
            float[][] depthMap = depthEstimator.estimateDepth(conv);
            assert conv != null;
            long startTime = System.currentTimeMillis();
            float[][] resizedDepthMap = resizeDepthMap(depthMap, Objects.requireNonNull(conv).getHeight(), conv.getWidth());
            long endTime = System.currentTimeMillis();
            System.out.println("Ridimensionamento eseguito in " + (endTime - startTime) + " millisecondi");
            ArrayList<Recognition> recognitions =  yolov5Detector.detect(conv);

            Bitmap mutableBitmap = conv.copy(Bitmap.Config.ARGB_8888, true);
            Canvas canvas = new Canvas(mutableBitmap);

            recognitionsInOrder = new ArrayList<>();

            for(Recognition recognition: recognitions){
                if(recognition.getConfidence() > 0.5){
                    RectF location = recognition.getLocation();
                    float centerX = location.centerX();
                    float centerY = location.centerY();
                    float depth = resizedDepthMap[(int) centerY][(int) centerX];
                    recognition.setDepth(depth);
                    recognitionsInOrder.add(recognition);
                }
            }
            //ordina gli elementi dal più vicino al più lontano dalla fotocamera
            Comparator<Recognition> comparator = Comparator.comparing(Recognition::getDepth).reversed();
            Collections.sort(recognitionsInOrder, comparator);

            for(int i = 0; i < recognitionsInOrder.size(); i++){
                Recognition recognition = recognitionsInOrder.get(i);
                RectF location = recognition.getLocation();
                canvas.drawRect(location, boxPaint);
                canvas.drawText(recognition.getLabelName() + ":" + i, location.left, location.top, textPain);
            }

            Drawable drawable = new BitmapDrawable(getResources(), mutableBitmap);
            this.pview.setForeground(drawable);
        }else {
            // Se analysis_on è false, visualizza solo l'immagine senza rilevamenti
            Drawable drawable = new BitmapDrawable(getResources(), conv);
            pview.setForeground(drawable);
        }
        image.close();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(getApplicationContext(), "Permission Granted", Toast.LENGTH_SHORT).show();

                // main logic
            } else {
                Toast.makeText(getApplicationContext(), "Permission Denied", Toast.LENGTH_SHORT).show();
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                        != PackageManager.PERMISSION_GRANTED) {
                    showMessageOKCancel(
                    );
                }
            }
        }
    }

    private void showMessageOKCancel() {
        new AlertDialog.Builder(MainActivity.this)
                .setMessage("You need to allow access permissions")
                .setPositiveButton("OK", null)
                .create()
                .show();
    }
}