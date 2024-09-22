#include <Arduino_APDS9960.h>

void setup() {
  Serial.begin(19200);
  while(!Serial);

  if (!APDS.begin())
    Serial.println("Error initializing APDS-9960 sensor!");
}

void loop() {
  if (APDS.proximityAvailable()) {
    int proximity = APDS.readProximity();

    Serial.println(proximity);
  }

  delay(10);
}
