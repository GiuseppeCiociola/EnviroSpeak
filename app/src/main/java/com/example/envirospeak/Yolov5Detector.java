package com.example.envirospeak;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;


import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect2d;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.opencv.dnn.Dnn;


public class Yolov5Detector {

    private final Size INPNUT_SIZE = new Size(320, 320);
    private final int[] OUTPUT_SIZE = new int[]{1, 6300, 85};

    private String MODEL_FILE;

    private Interpreter tflite;
    private List<String> associatedAxisLabels;
    Interpreter.Options options = new Interpreter.Options();


    public void setModelFile(String modelFile) {
        MODEL_FILE = modelFile;
    }

    public void initialModel(Context activity) {
        try {
            ByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, MODEL_FILE);
            tflite = new Interpreter(tfliteModel, options);

            String LABEL_FILE = "coco_label.txt";
            associatedAxisLabels = FileUtil.loadLabels(activity, LABEL_FILE);

        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading model or label: ", e);
            Toast.makeText(activity, "load model error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    public ArrayList<Recognition> detect(Bitmap bitmap) {

        int BITMAP_HEIGHT = bitmap.getHeight();
        int BITMAP_WIDTH = bitmap.getWidth();
        TensorImage yolov5sTfliteInput;
        ImageProcessor imageProcessor;
        imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(0, 255))
                        .build();
        yolov5sTfliteInput = new TensorImage(DataType.FLOAT32);

        yolov5sTfliteInput.load(bitmap);
        yolov5sTfliteInput = imageProcessor.process(yolov5sTfliteInput);


        TensorBuffer yoloOutput;
        yoloOutput = TensorBuffer.createFixedSize(OUTPUT_SIZE, DataType.FLOAT32);

        if (null != tflite)
            tflite.run(yolov5sTfliteInput.getBuffer(), yoloOutput.getBuffer());
        float[] recognitionArray = yoloOutput.getFloatArray();
        ArrayList<Recognition> allRecognitions = new ArrayList<>();
        for (int i = 0; i < OUTPUT_SIZE[1]; i++) {
            int gridStride = i * OUTPUT_SIZE[2];
            float x = recognitionArray[gridStride] * BITMAP_WIDTH;
            float y = recognitionArray[1 + gridStride] * BITMAP_HEIGHT;
            float w = recognitionArray[2 + gridStride] * BITMAP_WIDTH;
            float h = recognitionArray[3 + gridStride] * BITMAP_HEIGHT;
            int xmin = (int) Math.max(0, x - w / 2.);
            int ymin = (int) Math.max(0, y - h / 2.);
            int xmax = (int) Math.min(BITMAP_WIDTH, x + w / 2.);
            int ymax = (int) Math.min(BITMAP_HEIGHT, y + h / 2.);
            float confidence = recognitionArray[4 + gridStride];
            float[] classScores = Arrays.copyOfRange(recognitionArray, 5 + gridStride, this.OUTPUT_SIZE[2] + gridStride);
            int labelId = 0;
            float maxLabelScores = 0.f;
            for (int j = 0; j < classScores.length; j++) {
                if (classScores[j] > maxLabelScores) {
                    maxLabelScores = classScores[j];
                    labelId = j;
                }
            }
            Recognition r = new Recognition(
                    labelId,
                    associatedAxisLabels.get(labelId),
                    confidence,
                    new RectF(xmin, ymin, xmax, ymax));
            allRecognitions.add(
                    r);
        }
        return nms(allRecognitions);
    }

    protected ArrayList<Recognition> nms(ArrayList<Recognition> allRecognitions) {
        ArrayList<Recognition> nmsRecognitions = new ArrayList<>();
        MatOfRect2d boxes = new MatOfRect2d();
        ArrayList<Rect2d> rectList = new ArrayList<>();
        MatOfFloat scores = new MatOfFloat();
        float[] scoresArray = new float[allRecognitions.size()];
        for (int i = 0; i < allRecognitions.size(); i++) {
            Recognition recognition = allRecognitions.get(i);
            RectF location = recognition.getLocation();
            rectList.add(new Rect2d(location.left, location.top, location.width(), location.height()));
            scoresArray[i] = recognition.getConfidence();
        }
        boxes.fromList(rectList);
        scores.fromArray(scoresArray);
        MatOfInt indices = new MatOfInt();
        Dnn.NMSBoxes(boxes, scores, 0.0f, 0.45f, indices);
        int[] indicesArray = indices.toArray();
        for (int index : indicesArray) {
            nmsRecognitions.add(allRecognitions.get(index));
        }

        return nmsRecognitions;
    }

}