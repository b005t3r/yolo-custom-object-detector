import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.onnxruntime.runner.OnnxRuntimeRunner;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;

public class ONNXTest {
    public static void main(String[] args) throws OrtException, IOException {
        final int imageWidth            = 416;
        final int imageHeight           = 416;
        final int gridSize              = 13;
        final int numClasses            = 1;
        final float confidenceThreshold = 0.4f;
        final float nmsThreshold        = 0.2f;

        // Path to the ONNX model file
        final File modelPath    = new File("/Users/booster/Documents/Python/TrainYOLOv3/checkpoint_ball_detection_tiny/model_123.onnx");
        final File imageDir     = new File("/Users/booster/Documents/Python/TrainYOLOv3/data_ball_detection/ball/images");
        final File outputDir    = new File("/Users/booster/Documents/Python/TrainYOLOv3/test_ball_detection_tiny_onnx.123/ball/images");

        try(OnnxRuntimeRunner onnxRuntimeRunner = OnnxRuntimeRunner.builder()
                .modelUri(modelPath.getAbsolutePath())
                .build()) {

            for(File imageFile : imageDir.listFiles()) {
                if(! imageFile.isFile() || ! imageFile.getName().toLowerCase().endsWith(".jpg"))
                    continue;

                BufferedImage image = ImageIO.read(imageFile);

                INDArray input = new NativeImageLoader(image.getHeight(), image.getWidth(), 3).asMatrix(image);

                ImagePreProcessingScaler imageScaler = new ImagePreProcessingScaler(0, 1);
                imageScaler.transform(input);

                Map<String,INDArray> inputs = new HashMap<>();
                inputs.put("input", input);

                long start = System.currentTimeMillis();
                Map<String, INDArray> result = onnxRuntimeRunner.exec(inputs);

                INDArray output = result.entrySet().iterator().next().getValue();
                float[] values = output.data().asFloat();

                List<YoloUtil.BoundingBox> boxes = YoloUtil.decode(values, imageWidth, imageHeight, gridSize, numClasses, confidenceThreshold, nmsThreshold);
                long duration = System.currentTimeMillis() - start;

                for(YoloUtil.BoundingBox box : boxes) {
                    Graphics2D g = image.createGraphics();
                    g.setColor(Color.MAGENTA);
                    g.setStroke(new BasicStroke(2));
                    g.draw(box.createRectangle2D());
                    g.dispose();
                }

                System.out.println("Duration: " + duration);

                File outputFile = new File(imageFile.getAbsolutePath().replace(imageDir.getAbsolutePath(), outputDir.getAbsolutePath()));

                outputFile.getParentFile().mkdirs();
                ImageIO.write(image, "jpg", outputFile);
            }
        }
    }

    private static int[] findTopConfidenceLevels(float[] values, int maxNum) {
        float[] topValues = new float[maxNum];
        int[] topIndexes = new int[maxNum];

        Arrays.fill(topValues, Float.NEGATIVE_INFINITY);
        Arrays.fill(topIndexes, -1);

        for (int i = 0; i < values.length; i += 6) {
            for (int j = 0; j < topValues.length; j++) {
                if (values[i + 4] > topValues[j]) {

                    // Shift values and indexes down to make room for the new value and index
                    for (int k = topValues.length - 1; k > j; k--) {
                        topValues[k] = topValues[k - 1];
                        topIndexes[k] = topIndexes[k - 1];
                    }

                    topValues[j] = values[i + 4];
                    topIndexes[j] = i;
                    break;
                }
            }
        }

        return topIndexes;
    }
}
