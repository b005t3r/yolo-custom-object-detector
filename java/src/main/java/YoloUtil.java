import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.List;

public final class YoloUtil {
    public static class BoundingBox {
        float x1, y1, x2, y2, confidence;
        int classId;

        public BoundingBox(float x1, float y1, float x2, float y2, float confidence, int classId) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.confidence = confidence;
            this.classId = classId;
        }

        public Rectangle2D createRectangle2D() {
            return new Rectangle2D.Float(x1, y1, x2 - x1,y2 - y1);
        }

        @Override
        public String toString() {
            return String.format("%d: %.2f, [%.2f, %.2f, %.2f, %.2f]", classId, confidence, x1, y1, x2, y2);
        }
    }

    public static List<BoundingBox> decode(float[] output, int imageWidth, int imageHeight, int gridSize, int numClasses, float confidenceThreshold, float nmsThreshold) {
        List<BoundingBox> boxes = new ArrayList<>();
        int boxSize = 5 + numClasses; // x, y, w, h, confidence, class probabilities
        //int numCells = gridSize * gridSize;
        int numCells = output.length / boxSize;

        for (int i = 0; i < numCells; i++) {
            int offset = i * boxSize;
            float xCenter = output[offset]/* * imageWidth*/;
            float yCenter = output[offset + 1]/* * imageHeight*/;
            float width = output[offset + 2]/* * imageWidth*/;
            float height = output[offset + 3]/* * imageHeight*/;
            float confidence = output[offset + 4];

            if (confidence < confidenceThreshold) {
                continue;
            }

            float maxClassScore = -Float.MAX_VALUE;
            int classId = -1;
            for (int c = 0; c < numClasses; c++) {
                float classScore = output[offset + 5 + c];
                if (classScore > maxClassScore) {
                    maxClassScore = classScore;
                    classId = c;
                }
            }

            float x1 = xCenter - width / 2;
            float y1 = yCenter - height / 2;
            float x2 = xCenter + width / 2;
            float y2 = yCenter + height / 2;

            boxes.add(new BoundingBox(x1, y1, x2, y2, confidence * maxClassScore, classId));
        }

        return nonMaximumSuppression(boxes, nmsThreshold);
    }

    private static List<BoundingBox> nonMaximumSuppression(List<BoundingBox> boxes, float nmsThreshold) {
        List<BoundingBox> result = new ArrayList<>();

        boxes.sort((a, b) -> Float.compare(b.confidence, a.confidence));

        while (!boxes.isEmpty()) {
            BoundingBox best = boxes.remove(0);
            result.add(best);

            boxes.removeIf(box -> iou(best, box) > nmsThreshold);
        }

        return result;
    }

    private static float iou(BoundingBox box1, BoundingBox box2) {
        float interArea = intersectionArea(box1, box2);
        float unionArea = (box1.x2 - box1.x1) * (box1.y2 - box1.y1) +
                (box2.x2 - box2.x1) * (box2.y2 - box2.y1) - interArea;
        return interArea / unionArea;
    }

    private static float intersectionArea(BoundingBox box1, BoundingBox box2) {
        float x1 = Math.max(box1.x1, box2.x1);
        float y1 = Math.max(box1.y1, box2.y1);
        float x2 = Math.min(box1.x2, box2.x2);
        float y2 = Math.min(box1.y2, box2.y2);

        float width = Math.max(0, x2 - x1);
        float height = Math.max(0, y2 - y1);

        return width * height;
    }
}
