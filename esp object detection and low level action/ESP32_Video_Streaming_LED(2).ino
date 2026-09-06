#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "esp_http_server.h"

// ============================================================
// Wi-Fi settings
// ============================================================

const char *WIFI_SSID = "Your wifi name";
const char *WIFI_PASSWORD = "Your wifi password";

// ============================================================
// GPIO settings
// ============================================================

// GPIO13 becomes HIGH when Python detects a person.
constexpr int GPIO_PERSON_LED = 13;

// GPIO12 always remains LOW.
constexpr int GPIO_ALWAYS_LOW = 12;

// ============================================================
// AI Thinker ESP32-CAM pin mapping
// ============================================================

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5

#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22


// Two separate servers:
// Port 80 = GPIO control
// Port 81 = MJPEG camera stream
httpd_handle_t controlServer = nullptr;
httpd_handle_t streamServer = nullptr;


// ============================================================
// GPIO control
// ============================================================

void setPersonOutput(bool personDetected) {
  digitalWrite(GPIO_PERSON_LED, personDetected ? HIGH : LOW);
  digitalWrite(GPIO_ALWAYS_LOW, LOW);

  if (personDetected) {
    Serial.println("GPIO13 HIGH: person detected");
  } else {
    Serial.println("GPIO13 LOW: no person");
  }
}

void sendJson(httpd_req_t *request, const char *json) {
  httpd_resp_set_type(request, "application/json");
  httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", "*");
  httpd_resp_sendstr(request, json);
}


// ============================================================
// GET /person_on
//
// Python calls this when YOLO confirms a person.
// GPIO13 HIGH, GPIO12 LOW.
// ============================================================

esp_err_t personOnHandler(httpd_req_t *request) {
  setPersonOutput(true);

  sendJson(
    request,
    "{\"ok\":true,\"gpio13\":\"HIGH\",\"gpio12\":\"LOW\"}"
  );

  return ESP_OK;
}


// ============================================================
// GET /person_off
//
// Python calls this when YOLO confirms no person.
// GPIO13 LOW, GPIO12 LOW.
// ============================================================

esp_err_t personOffHandler(httpd_req_t *request) {
  setPersonOutput(false);

  sendJson(
    request,
    "{\"ok\":true,\"gpio13\":\"LOW\",\"gpio12\":\"LOW\"}"
  );

  return ESP_OK;
}


// ============================================================
// GET :81/stream
//
// Continuously sends ESP32-CAM JPEG frames to the PC.
// This is on port 81, so it cannot block GPIO commands.
// ============================================================

esp_err_t streamHandler(httpd_req_t *request) {
  esp_err_t result = httpd_resp_set_type(
    request,
    "multipart/x-mixed-replace;boundary=frame"
  );

  while (result == ESP_OK) {
    camera_fb_t *frame = esp_camera_fb_get();

    if (!frame) {
      Serial.println("Camera capture failed");
      return ESP_FAIL;
    }

    char header[100];

    int headerLength = snprintf(
      header,
      sizeof(header),
      "--frame\r\n"
      "Content-Type: image/jpeg\r\n"
      "Content-Length: %u\r\n\r\n",
      frame->len
    );

    result = httpd_resp_send_chunk(
      request,
      header,
      headerLength
    );

    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(
        request,
        reinterpret_cast<const char *>(frame->buf),
        frame->len
      );
    }

    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(request, "\r\n", 2);
    }

    esp_camera_fb_return(frame);
  }

  return result;
}


// ============================================================
// Starts both HTTP servers
// ============================================================

void startHttpServers() {
  // ----------------------------------------------------------
  // Control server: port 80
  // Handles /person_on and /person_off
  // ----------------------------------------------------------
  httpd_config_t controlConfig = HTTPD_DEFAULT_CONFIG();

  controlConfig.server_port = 80;
  controlConfig.ctrl_port = 32768;

  httpd_uri_t personOnRoute = {};
  personOnRoute.uri = "/person_on";
  personOnRoute.method = HTTP_GET;
  personOnRoute.handler = personOnHandler;

  httpd_uri_t personOffRoute = {};
  personOffRoute.uri = "/person_off";
  personOffRoute.method = HTTP_GET;
  personOffRoute.handler = personOffHandler;

  if (httpd_start(&controlServer, &controlConfig) == ESP_OK) {
    httpd_register_uri_handler(controlServer, &personOnRoute);
    httpd_register_uri_handler(controlServer, &personOffRoute);

    Serial.println("GPIO control server started on port 80");
  } else {
    Serial.println("GPIO control server failed");
  }

  // ----------------------------------------------------------
  // Stream server: port 81
  // Handles /stream only
  // ----------------------------------------------------------
  httpd_config_t streamConfig = HTTPD_DEFAULT_CONFIG();

  streamConfig.server_port = 81;
  streamConfig.ctrl_port = 32769;

  httpd_uri_t streamRoute = {};
  streamRoute.uri = "/stream";
  streamRoute.method = HTTP_GET;
  streamRoute.handler = streamHandler;

  if (httpd_start(&streamServer, &streamConfig) == ESP_OK) {
    httpd_register_uri_handler(streamServer, &streamRoute);

    Serial.println("Camera stream server started on port 81");
  } else {
    Serial.println("Camera stream server failed");
  }
}


// ============================================================
// Setup
// ============================================================

void setup() {
  Serial.begin(115200);

  // Start both pins LOW for safety.
  pinMode(GPIO_PERSON_LED, OUTPUT);
  pinMode(GPIO_ALWAYS_LOW, OUTPUT);
  setPersonOutput(false);

  camera_config_t cameraConfig = {};

  cameraConfig.ledc_channel = LEDC_CHANNEL_0;
  cameraConfig.ledc_timer = LEDC_TIMER_0;

  cameraConfig.pin_d0 = Y2_GPIO_NUM;
  cameraConfig.pin_d1 = Y3_GPIO_NUM;
  cameraConfig.pin_d2 = Y4_GPIO_NUM;
  cameraConfig.pin_d3 = Y5_GPIO_NUM;
  cameraConfig.pin_d4 = Y6_GPIO_NUM;
  cameraConfig.pin_d5 = Y7_GPIO_NUM;
  cameraConfig.pin_d6 = Y8_GPIO_NUM;
  cameraConfig.pin_d7 = Y9_GPIO_NUM;

  cameraConfig.pin_xclk = XCLK_GPIO_NUM;
  cameraConfig.pin_pclk = PCLK_GPIO_NUM;
  cameraConfig.pin_vsync = VSYNC_GPIO_NUM;
  cameraConfig.pin_href = HREF_GPIO_NUM;

  cameraConfig.pin_sccb_sda = SIOD_GPIO_NUM;
  cameraConfig.pin_sccb_scl = SIOC_GPIO_NUM;

  cameraConfig.pin_pwdn = PWDN_GPIO_NUM;
  cameraConfig.pin_reset = RESET_GPIO_NUM;

  cameraConfig.xclk_freq_hz = 20000000;
  cameraConfig.pixel_format = PIXFORMAT_JPEG;

  // QVGA is faster and suitable for CPU YOLO.
  cameraConfig.frame_size = FRAMESIZE_QVGA;
  cameraConfig.jpeg_quality = 15;
  cameraConfig.fb_count = 1;

  if (esp_camera_init(&cameraConfig) != ESP_OK) {
    Serial.println("Camera initialization failed");

    while (true) {
      delay(1000);
    }
  }

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to Wi-Fi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  startHttpServers();

  Serial.println();
  Serial.println("Camera stream:");
  Serial.println("http://192.168.0.172:81/stream");

  Serial.println("LED ON test:");
  Serial.println("http://192.168.0.172/person_on");

  Serial.println("LED OFF test:");
  Serial.println("http://192.168.0.172/person_off");
}

void loop() {
  // Both HTTP servers run automatically.
}
