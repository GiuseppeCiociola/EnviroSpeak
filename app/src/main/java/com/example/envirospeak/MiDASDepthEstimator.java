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

public class MiDASDepthEstimator {
    // Costanti per le dimensioni dell'immagine di input del modello MiDaS
    private final Size INPNUT_SIZE = new Size(256, 256);
    private final int[] OUTPUT_SIZE = new int[]{1, 256, 256, 1};
    private static final int INPUT_WIDTH = 256;
    private static final int INPUT_HEIGHT = 256;
    private final Interpreter interpreter;
    private final float[][][] outputArray;

    public MiDASDepthEstimator(Context context, String modelPath) throws IOException {
        // Carica il modello TensorFlow Lite dall'asset
        ByteBuffer modelBuffer = FileUtil.loadMappedFile(context, modelPath);
        //MappedByteBuffer modelBuffer = loadModelFile(context, modelPath);

        // Inizializza l'interprete TensorFlow Lite
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4); // Imposta il numero di thread per l'elaborazione parallela
        interpreter = new Interpreter(modelBuffer, options);
        outputArray = new float[1][INPUT_HEIGHT][INPUT_WIDTH];
    }

    // Metodo per eseguire l'inferenza del modello MiDaS e ottenere la mappa di profondità
    public float[][] estimateDepth(Bitmap inputImage) {
        // Converti la Bitmap in un ByteBuffer conforme al formato di input del modello
        TensorImage midasInput;
        ImageProcessor imageProcessor;
        imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(0, 255))
                        .build();
        midasInput = new TensorImage(DataType.FLOAT32);


        midasInput.load(inputImage);
        midasInput = imageProcessor.process(midasInput);

        ByteBuffer outputBuffer = TensorBuffer.createFixedSize(OUTPUT_SIZE, DataType.FLOAT32).getBuffer();

        // Esegui l'inferenza del modello
        long startTime = SystemClock.uptimeMillis();
        interpreter.run(midasInput.getBuffer(), outputBuffer);
        long endTime = SystemClock.uptimeMillis();
        System.out.println("Inferenza completata in " + (endTime - startTime) + " ms");

        // Ottieni i risultati dalla ByteBuffer di output
        outputBuffer.rewind();
        for (int i = 0; i < INPUT_HEIGHT; i++) {
            for (int j = 0; j < INPUT_WIDTH; j++) {
                outputArray[0][i][j] = outputBuffer.getFloat();
            }
        }

        return outputArray[0];
    }
}