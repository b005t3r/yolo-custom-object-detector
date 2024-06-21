import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.onnxruntime.runner.OnnxRuntimeRunner;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class ONNXTest {
    public static void main(String[] args) throws OrtException, IOException {
        // Path to the ONNX model file
        File modelPath = new File("/Users/booster/Documents/Python/TrainYOLOv3/checkpoint_ball_detection/model.onnx");

        // Assuming the model expects input of shape [1, 3, 416, 416]
        //INDArray input = Nd4j.rand(1, 3, 416, 416);

        //BufferedImage myImage = ImageIO.read(new File("/Users/booster/Documents/Python/TrainYOLOv3/data_ball_detection/ball/images/tournament_Untitled_0013_mirrored_14080733.jpg"));
        BufferedImage myImage = ImageIO.read(new File("/Users/booster/Documents/Python/TrainYOLOv3/data_ball_detection/ball/images/tournament_Untitled_0137_5005000.jpg"));

        NativeImageLoader loader = new NativeImageLoader(myImage.getHeight(), myImage.getWidth(), 3);
        INDArray input = loader.asMatrix(myImage);
        ImagePreProcessingScaler imageScaler = new ImagePreProcessingScaler(0, 1);

        imageScaler.transform(input);

        try(OnnxRuntimeRunner onnxRuntimeRunner = OnnxRuntimeRunner.builder()
                .modelUri(modelPath.getAbsolutePath())
                .build()) {

            Map<String,INDArray> inputs = new HashMap<>();
            inputs.put("input.1", input);

            long start = System.currentTimeMillis();
            Map<String, INDArray> result = onnxRuntimeRunner.exec(inputs);
            long duration = System.currentTimeMillis() - start;

            INDArray output = result.entrySet().iterator().next().getValue();
            float[] values = output.data().asFloat();

            int highest = 0;
            for(int i = 0; i < values.length; i += 6) {
                if(values[i + 4] > values[highest + 4])
                    highest = i;
            }

            System.out.printf(
                    "%.6f %.6f %.6f %.6f %.6f %.6f\n",
                    values[highest + 0], values[highest + 1], values[highest + 2], values[highest + 3], values[highest + 4], values[highest + 5]
            );


            /*
            for(int i = 0; i < values.length; i += 6) {
                System.out.printf(
                    "%.6f %.6f %.6f %.6f %.6f %.6f\n",
                    values[i + 0], values[i + 1], values[i + 2], values[i + 3], values[i + 4], values[i + 5]
                );
            }
*/

            System.out.println(output);
            System.out.println("Duration: " + duration);
        }
    }
}
