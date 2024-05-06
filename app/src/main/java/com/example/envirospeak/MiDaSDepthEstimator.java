package com.example.envirospeak;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Size;

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

public class MiDaSDepthEstimator {
    // Costanti per le dimensioni dell'immagine di input del modello MiDaS
    private final Size INPNUT_SIZE = new Size(256, 256);
    private final int[] OUTPUT_SIZE = new int[]{1, 256, 256, 1};
    private static final int INPUT_WIDTH = 256;
    private static final int INPUT_HEIGHT = 256;
    private final Interpreter tflite;
    private final float[][] outputArray;

    public MiDaSDepthEstimator(Context context, String modelPath) throws IOException {
        ByteBuffer modelBuffer = FileUtil.loadMappedFile(context, modelPath);
        Interpreter.Options options = new Interpreter.Options();
        tflite = new Interpreter(modelBuffer, options);
        outputArray = new float[INPUT_HEIGHT][INPUT_WIDTH];
    }

    public float[][] estimateDepth(Bitmap bitmap) {
        TensorImage midasInput;
        ImageProcessor imageProcessor;
        imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(0, 255))
                        .build();
        midasInput = new TensorImage(DataType.FLOAT32);

        midasInput.load(bitmap);
        midasInput = imageProcessor.process(midasInput);

        ByteBuffer outputBuffer = TensorBuffer.createFixedSize(OUTPUT_SIZE, DataType.FLOAT32).getBuffer();

        if (null != tflite)
            tflite.run(midasInput.getBuffer(), outputBuffer);

        outputBuffer.rewind();
        for (int i = 0; i < INPUT_HEIGHT; i++) {
            for (int j = 0; j < INPUT_WIDTH; j++) {
                outputArray[i][j] = outputBuffer.getFloat();
            }
        }

        return outputArray;
    }
}