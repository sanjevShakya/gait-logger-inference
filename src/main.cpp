#include <Arduino.h>
#include <Arduino_LSM9DS1.h>
#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>

// sine_model.h contains the array you exported from Python with xxd or tinymlgen
// #include "sine_model.h"
#include "model.h"

// acceleration parameters
constexpr int window_length = 128;
constexpr int num_features = 3;
constexpr int num_y_labels = 3;
int acceleration_index_counter = 0;
bool is_initial_tensor_filled = false;
constexpr int overlap_window_length = 64;
float accel_tensor[128][3] = {};
float overlap_accel_tensor[128][3] = {};

#define N_INPUTS window_length *num_features
#define N_OUTPUTS num_y_labels
// it's a trial and error process
#define TENSOR_ARENA_SIZE 20 * 1024

Eloquent::TinyML::TensorFlow::TensorFlow<N_INPUTS, N_OUTPUTS, TENSOR_ARENA_SIZE> tf;

float X_test[N_INPUTS] = {};

uint8_t y_test[3] = {0};

void handle_read_acceleration();

void setup()
{
  Serial.begin(115200);
  delay(4000);
  while (!Serial)
    ;
  if (!IMU.begin())
  {
    Serial.println("Failed to initialize IMU!");
    while (1)
      ;
  }
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.println();
  Serial.println("Acceleration in g's");
  Serial.println("X\tY\tZ");

  Serial.println("----Loading TF model----");
  tf.begin(model_data);

  // check if model loaded fine
  if (!tf.isOk())
  {
    Serial.print("ERROR: ");
    Serial.println(tf.getErrorMessage());

    while (true)
      delay(1000);
  }
  Serial.println("----TF model successfuly loaded----");
}

void loop()
{
  handle_read_acceleration();
}

void handle_read_acceleration()
{
  if (IMU.accelerationAvailable())
  {
    const int acceleration_index = (acceleration_index_counter % window_length);
    // fill the bucket
    float *current_accleration_data = accel_tensor[acceleration_index];
    if (!IMU.readAcceleration(current_accleration_data[0], current_accleration_data[1], current_accleration_data[2]))
    {
      Serial.println("Error reading acceleration data");
      return;
    }
    acceleration_index_counter++;

    if (acceleration_index == window_length - 1)
    {
      // inference here
      // copy the accel tensor to overlap tensor
      // copy the lower half of accel tensor to front half of
      // accel tensor and reset the window lenght to overlap length12

      for (int i = overlap_window_length; i < window_length; i++)
      {
        float *accel_data = overlap_accel_tensor[overlap_window_length - i];
        overlap_accel_tensor[i][0] = accel_data[0];
        overlap_accel_tensor[i][1] = accel_data[1];
        overlap_accel_tensor[i][2] = accel_data[2];
      }

      for (int i = 0; i < overlap_window_length; i++)
      {
        float *accel_data = accel_tensor[i + overlap_window_length];
        overlap_accel_tensor[i][0] = accel_data[0];
        overlap_accel_tensor[i][1] = accel_data[1];
        overlap_accel_tensor[i][2] = accel_data[2];
      }
      for (int i = 0; i < window_length; i++)
      {
        float *accel_data = overlap_accel_tensor[i];
        X_test[i] = accel_data[0];
        X_test[i + 1] = accel_data[1];
        X_test[i + 2] = accel_data[2];
      }
      Serial.print("Predicted Label = ");
      Serial.println(tf.predictClass(X_test));
      // delay(100);
    }
  }
}