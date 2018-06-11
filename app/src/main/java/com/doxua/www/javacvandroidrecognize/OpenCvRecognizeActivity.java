package com.doxua.www.javacvandroidrecognize;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Size;
import java.io.File;
import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_PLAIN;
import static org.bytedeco.javacpp.opencv_core.LINE_8;
import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_face.EigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import static com.doxua.www.javacvandroidrecognize.TrainHelper.ACCEPT_LEVEL;

public class OpenCvRecognizeActivity extends Activity implements CvCameraPreview.CvCameraViewListener {

    public static final String TAG = "OpenCvRecognizeActivity";

    private CascadeClassifier faceDetector;
    private String[] nomes = {"", "Y Know You"};
    private int absoluteFaceSize = 0;
    private CvCameraPreview cameraView;
    boolean takePhoto;
    opencv_face.FaceRecognizer faceRecognizer = EigenFaceRecognizer.create();
    boolean trained;


    private static final int PICK_IMAGE = 100;
    private ImageView imageView;

    private boolean hasPermissions(Context context, String... permissions) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && context != null && permissions != null) {
            for (String permission : permissions) {
                if (ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                    return false;
                }
            }
        }
        return true;
    }

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_opencv);
        if (Build.VERSION.SDK_INT >= 23) {
            String[] PERMISSIONS = {android.Manifest.permission.READ_EXTERNAL_STORAGE,android.Manifest.permission.WRITE_EXTERNAL_STORAGE};
            if (!hasPermissions(this, PERMISSIONS)) {
                ActivityCompat.requestPermissions(this, PERMISSIONS, 1 );
            }
        }

        // Call the camera view to display the camera screen.
        cameraView = (CvCameraPreview) findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);


        // Call the image view tp display the selected image from the gallery.
        imageView = (ImageView) findViewById(R.id.image_view);
        Button pickImageButton = (Button) findViewById(R.id.pick_image_button);
        pickImageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openGallery();
            }
        });



        new AsyncTask<Void,Void,Void>() {


            // Detect the face and save the face detected to the path for face recognition later on.
            @Override
            protected Void doInBackground(Void... voids) {
                try {

                    // Load the CascadeClassifier class to detect objects in a video stream. Particularly, we will use the functions:
                    // load to load a .xml classifier file. It can be either a Haar or a LBP classifer
                    // detectMultiScale to perform the detection.
                    // TO-DO: Detect faces in an image.
                    faceDetector = TrainHelper.loadClassifierCascade(OpenCvRecognizeActivity.this, R.raw.frontalface);




                    if (TrainHelper.isTrained(getBaseContext())) {


                        // Create a new folder named 'train_folder'.
                        // Create a new file named 'eigenFacesClassifier.yml' inside the folder above.
                        File folder = new File(getFilesDir(), TrainHelper.TRAIN_FOLDER);
                        File f = new File(folder, TrainHelper.EIGEN_FACES_CLASSIFIER);


                        // Loads a persisted model and state from a given XML or YAML file.
                        // Every FaceRecognizer has to overwrite FaceRecognizer::load(FileStorage& fs) to enable loading the model state.
                        faceRecognizer.read(f.getAbsolutePath());




                        trained = true;
                    }
                } catch (Exception e) {
                    Log.d(TAG, e.getLocalizedMessage(), e);
                }
                return null;
            }

            // When users click on the selected button.
            // Active the correct action to implement.
            @Override
            protected void onPostExecute(Void aVoid) {
                super.onPostExecute(aVoid);
                findViewById(R.id.btPhoto).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        takePhoto = true; // Execute the capturePhoto on the onCameraFrame.
                    }
                });
                findViewById(R.id.btTrain).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        train();
                    }
                });
                findViewById(R.id.btReset).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        try {
                            TrainHelper.reset(getBaseContext());
                            Toast.makeText(getBaseContext(), "Reset with success.", Toast.LENGTH_SHORT).show();
                            finish();
                        } catch (Exception e) {
                            Log.d(TAG, e.getLocalizedMessage(), e);
                        }
                    }
                });
            }
        }.execute();
    }


    // Add-on method
    private void openGallery() {
        Intent gallery =
                new Intent(Intent.ACTION_PICK,
                        android.provider.MediaStore.Images.Media.INTERNAL_CONTENT_URI);
        startActivityForResult(gallery, PICK_IMAGE);
    }


    // Add-on method
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == PICK_IMAGE) {
            Uri imageUri = data.getData();
            imageView.setImageURI(imageUri);
        }
    }

    void train() {
        int remainingPhotos = TrainHelper.PHOTOS_TRAIN_QTY - TrainHelper.qtdPhotos(getBaseContext());
        if (remainingPhotos > 0) {
            Toast.makeText(getBaseContext(), "You need more to call train: "+ remainingPhotos, Toast.LENGTH_SHORT).show();
            return;
        } else if (TrainHelper.isTrained(getBaseContext())) {
            Toast.makeText(getBaseContext(), "Already trained", Toast.LENGTH_SHORT).show();
            return;
        }

        Toast.makeText(getBaseContext(), "Start train: ", Toast.LENGTH_SHORT).show();
        new AsyncTask<Void, Void, Void>() {

            @Override
            protected Void doInBackground(Void... voids) {
                try {
                    if(!TrainHelper.isTrained(getBaseContext())) {
                        TrainHelper.train(getBaseContext());
                    }
                } catch (Exception e) {
                    Log.d(TAG, e.getLocalizedMessage(), e);
                }
                return null;
            }

            @Override
            protected void onPostExecute(Void aVoid) {
                super.onPostExecute(aVoid);
                try {
                    Toast.makeText(getBaseContext(), "Resetting after train - Success : "+ TrainHelper.isTrained(getBaseContext()), Toast.LENGTH_SHORT).show();
                    finish();
                } catch (Exception e) {
                    Log.d(TAG, e.getLocalizedMessage(), e);
                }
            }
        }.execute();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        absoluteFaceSize = (int) (width * 0.32f);
    }

    @Override
    public void onCameraViewStopped() {

    }


    // Capture the photo from the camera.
    private void capturePhoto(Mat rgbaMat) {
        try {
            TrainHelper.takePhoto(getBaseContext(), 1, TrainHelper.qtdPhotos(getBaseContext()) + 1, rgbaMat.clone(), faceDetector);
        } catch (Exception e) {
            e.printStackTrace();
        }
        takePhoto = false;
    }


    // Recognize the person's face.
    private void recognize(opencv_core.Rect dadosFace, Mat grayMat, Mat rgbaMat) {
        Mat detectedFace = new Mat(grayMat, dadosFace);
        resize(detectedFace, detectedFace, new Size(TrainHelper.IMG_SIZE,TrainHelper.IMG_SIZE));

        IntPointer label = new IntPointer(1);
        DoublePointer reliability = new DoublePointer(1);
        faceRecognizer.predict(detectedFace, label, reliability);
        int prediction = label.get(0);
        double acceptanceLevel = reliability.get(0);
        String name;

        // If there is not match calling 'Unknown'.
        if (prediction == -1 || acceptanceLevel >= ACCEPT_LEVEL) {
            name = getString(R.string.unknown);
        }
        // If there is match, display the percentage of matching.
        else {
            name = nomes[prediction] + " - " + acceptanceLevel;
        }
        int x = Math.max(dadosFace.tl().x() - 10, 0);
        int y = Math.max(dadosFace.tl().y() - 10, 0);

        // Display the text on the screen.
        putText(rgbaMat, name, new Point(x, y), FONT_HERSHEY_PLAIN, 1.4, new opencv_core.Scalar(0,255,0,0));
    }


    // Show the detected rectangle on the face.
    void showDetectedFace(RectVector faces, Mat rgbaMat) {
        int x = faces.get(0).x();
        int y = faces.get(0).y();
        int w = faces.get(0).width();
        int h = faces.get(0).height();

        rectangle(rgbaMat, new Point(x, y), new Point(x + w, y + h), opencv_core.Scalar.GREEN, 2, LINE_8, 0);
    }

    void noTrainedLabel(opencv_core.Rect face, Mat rgbaMat) {
        int x = Math.max(face.tl().x() - 10, 0);
        int y = Math.max(face.tl().y() - 10, 0);
        putText(rgbaMat, "No trained or train unavailable", new Point(x, y), FONT_HERSHEY_PLAIN, 1.4, new opencv_core.Scalar(0,255,0,0));
    }

    // Display the camera screen.
    // This method is override the class CvCameraPreview.java
    @Override
    public Mat onCameraFrame(Mat rgbaMat) {
        if (faceDetector != null) {
            Mat greyMat = new Mat(rgbaMat.rows(), rgbaMat.cols());
            cvtColor(rgbaMat, greyMat, CV_BGR2GRAY);
            RectVector faces = new RectVector();
            faceDetector.detectMultiScale(greyMat, faces, 1.25f, 3, 1,
                    new Size(absoluteFaceSize, absoluteFaceSize),
                    new Size(4 * absoluteFaceSize, 4 * absoluteFaceSize));

            if (faces.size() == 1) {
                showDetectedFace(faces, rgbaMat);
                if (takePhoto) {
                    capturePhoto(rgbaMat);
                    alertRemainingPhotos();
                }
                if (trained) {
                    recognize(faces.get(0), greyMat, rgbaMat);
                } else {
                    noTrainedLabel(faces.get(0), rgbaMat);
                }
            }
            greyMat.release();
        }
        return rgbaMat;
    }

    // Display the message.
    void alertRemainingPhotos() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                int remainingPhotos = TrainHelper.PHOTOS_TRAIN_QTY - TrainHelper.qtdPhotos(getBaseContext());
                Toast.makeText(getBaseContext(), "You need more to call train: "+ remainingPhotos, Toast.LENGTH_SHORT).show();
            }
        });
    }
}
