package com.emaraic.rubikcubedetector;

import com.emaraic.utils.YoloModel;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Random;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.profiler.ProfilerConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created on Jul 4, 2018 , 11:07:52 AM
 *
 * @author Taha Emara 
 * Email : taha@emaraic.com 
 * Website: http://www.emaraic.com
 */
public class YoloTrainer {

    private static final Logger log = LoggerFactory.getLogger(YoloTrainer.class);

    private static final int INPUT_WIDTH = 416;
    private static final int INPUT_HEIGHT = 416;
    private static final int CHANNELS = 3;

    private static final int GRID_WIDTH = 13;
    private static final int GRID_HEIGHT = 13;
    private static final int CLASSES_NUMBER = 1;
    private static final int BOXES_NUMBER = 5;
    private static final double[][] PRIOR_BOXES = {{1.5, 1.5}, {2, 2}, {3, 3}, {3.5, 8}, {4, 9}};

    private static final int BATCH_SIZE = 4;
    private static final int EPOCHS = 5; // 50;
    private static final double LEARNING_RATE = 0.001; // 0.0001;
    private static final int SEED = 12345;

    /*parent Dataset folder "DATA_DIR" contains two subfolder "images" and "annotations" */
    private static final String DATA_DIR = "Dataset";

    /* Yolo loss function prameters for more info
    https://stats.stackexchange.com/questions/287486/yolo-loss-function-explanation*/
    private static final double LAMDBA_COORD = 1.0f; // 1.0;
    private static final double LAMDBA_NO_OBJECT = 0.5;

    public static void main(String[] args) throws IOException, InterruptedException {
//        System.setProperty("org.bytedeco.javacpp.noPointerGC", "true");
//        System.setProperty("org.bytedeco.javacpp.maxBytes", "12G");
//        System.setProperty("org.bytedeco.javacpp.maxPhysicalBytes", "20G");

        try(PointerScope scope = new PointerScope()) {

            Nd4j.getExecutioner().setProfilingConfig(ProfilerConfig.builder()
                    .checkForINF(true)
                    .checkForNAN(true)
                    .checkElapsedTime(true)
                    .checkLocality(true)
                    .checkWorkspaces(true)
                    .build());

            Random rng = new Random(SEED);

/*
            //Initialize the user interface backend, it is just as tensorboard.
            //it starts at http://localhost:9000
            UIServer uiServer = UIServer.getInstance();

            //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
            StatsStorage statsStorage = new InMemoryStatsStorage();

            //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
            uiServer.attach(statsStorage);
*/

            File imageDir = new File(DATA_DIR, "images");

            log.info("Load data...");
            RandomPathFilter pathFilter = new RandomPathFilter(rng) {
                @Override
                protected boolean accept(String name) {
                    name = name.replace("/images/", "/annotations/").replace(".jpg", ".xml");
                    //System.out.println("Name " + name);
                    try {
                        return new File(new URI(name)).exists();
                    } catch (URISyntaxException ex) {
                        throw new RuntimeException(ex);
                    }
                }
            };

            InputSplit[] data = new FileSplit(imageDir, NativeImageLoader.ALLOWED_FORMATS, rng).sample(pathFilter, 0.9, 0.1);
            InputSplit trainData = data[0];
            InputSplit testData = data[1];

            ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(INPUT_HEIGHT, INPUT_WIDTH, CHANNELS,
                    GRID_HEIGHT, GRID_WIDTH, new VocLabelProvider(DATA_DIR));
            recordReaderTrain.initialize(trainData);

            ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(INPUT_HEIGHT, INPUT_WIDTH, CHANNELS,
                    GRID_HEIGHT, GRID_WIDTH, new VocLabelProvider(DATA_DIR));
            recordReaderTest.initialize(testData);

            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1, 8);
            //scaler.fitLabel(true);

            RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, BATCH_SIZE, 1, 1, true);
            train.setPreProcessor(scaler);

            RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
            test.setPreProcessor(scaler);

        ComputationGraph pretrained = (ComputationGraph) TinyYOLO.builder().build().initPretrained();
        INDArray priors = Nd4j.create(PRIOR_BOXES);

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .seed(SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder().learningRate(LEARNING_RATE).build())
                //.updater(new Nesterovs.Builder().learningRate(learningRate).momentum(lrMomentum).build())
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();

        ComputationGraph model = new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .removeVertexKeepConnections("conv2d_9")
                .removeVertexKeepConnections("outputs")
                .addLayer("convolution2d_9",
                        new ConvolutionLayer.Builder(1,1)
                                .nIn(1024)
                                .nOut(BOXES_NUMBER * (5 + CLASSES_NUMBER))
                                .stride(1,1)
                                .convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.UNIFORM)
                                .hasBias(false)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "leaky_re_lu_8")
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .lambdaNoObj(LAMDBA_NO_OBJECT)
                                .lambdaCoord(LAMDBA_COORD)
                                .boundingBoxPriors(priors)
                                .build(),
                        "convolution2d_9")
                .setOutputs("outputs")
                .build();

        System.out.println(model.summary(InputType.convolutional(INPUT_HEIGHT, INPUT_WIDTH, CHANNELS)));

            log.info("Train model...");
            model.setListeners(new ScoreIterationListener(1));//print score after each iteration on stout
            //model.setListeners(new StatsListener(statsStorage));// visit http://localhost:9000 to track the training process
            for (int i = 0; i < EPOCHS; i++) {
                train.reset();
                while (train.hasNext()) {
                    DataSet ds = train.next();

/*
                    if (ds.toString().contains("NaN")) {
                        System.err.println("NaN present");

                        System.out.println(ds);
                    }
*/

                    model.fit(ds);
                }
                log.info("*** Completed epoch {} ***", i);
            }
            //model.fit(train, EPOCHS);

            log.info("*** Saving Model ***");
            ModelSerializer.writeModel(model, "model.data", true);
            log.info("*** Training Done ***");


            //visualize results on the test set, Just hit any key in your keyboard to iterate the test set.
            log.info("*** Visualizing model on test data ***");
            YoloModel detector = new YoloModel();
            CanvasFrame mainframe = new CanvasFrame("Rubix Cube");
            mainframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
            mainframe.setCanvasSize(600, 600);
            mainframe.setLocationRelativeTo(null);
            mainframe.setVisible(true);

            OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
            test.setCollectMetaData(true);
            while (test.hasNext() && mainframe.isVisible()) {
                org.nd4j.linalg.dataset.DataSet ds = test.next();
                RecordMetaDataImageURI metadata = (RecordMetaDataImageURI) ds.getExampleMetaData().get(0);
                Mat image = imread(metadata.getURI().getPath());
                //System.out.println("Path: " +metadata.getURI().getPath());
                //detector.detectRubixCube(ds.getFeatures(), image, 0.4);
                detector.detectRubixCube(image, 0.1);
                mainframe.setTitle(new File(metadata.getURI()).getName());
                mainframe.showImage(converter.convert(image));
                mainframe.waitKey();
            }
            mainframe.dispose();
            System.exit(0);
        }
    }
}
