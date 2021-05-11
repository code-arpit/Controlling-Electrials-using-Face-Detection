#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

//set wifi credentials
#define WIFI_USERNAME "Arihant"
#define WIFI_PASSWORD "arpit13579"
#define UDP_PORT 4210

// UDP
WiFiUDP UDP;
char reply[] = "packet received";


void setup (){
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(D6, OUTPUT);
  pinMode(D7, OUTPUT);
  pinMode(D8, INPUT);
  pinMode(D4, OUTPUT);
  //setup serial 
  Serial.begin(115200);
  Serial.println();

  //Begin WIFI
  WiFi.begin(WIFI_USERNAME , WIFI_PASSWORD);

  //connecting to wifi...
  Serial.print("Connecting to .....");
  Serial.print(WIFI_USERNAME);

  //loop continue until wifi is connected
  while (WiFi.status() != WL_CONNECTED){
    delay(1000);
    Serial.println("Trying...");  
  }
  
  //after connection
  Serial.println();
  Serial.print("Connected! IP Address:\t");
  Serial.println(WiFi.localIP());

  //begin listening to UDP port
  UDP.begin(UDP_PORT);
  Serial.print("listening to udp port:\t");
  Serial.println(UDP_PORT);
  
  
}

void loop(){

  while(1){
    //if packet is received
    int p = UDP.parsePacket();
    if (p>0){ 
      Serial.print("Received packet! Size: ");
      Serial.println(p);
        char rdata[p+1];
        UDP.read(rdata,p);
        rdata[p] = '\0';
        Serial.print("packet Received:");
        Serial.println(rdata);
        r_p(String(rdata));      
        
      /*reply to received packet
        UDP.beginPacket(UDP.remoteIP(), UDP.remotePort());
        UDP.write(reply);
        UDP.endPacket(); */
          
    }       
  }
}

void r_p(String a){
  
  int i = 0;
  int r_p = a.toInt();
  if (r_p == 0) {
        digitalWrite(LED_BUILTIN, LOW);
  }
  else if (r_p  == 1) {
      digitalWrite(D6, HIGH);
      digitalWrite(D7, LOW);
      motion(r_p);
    
  }
  /*else if (r_p ==2){
       
       digitalWrite(LED_BUILTIN, LOW);
        delay(1000);
        digitalWrite(LED_BUILTIN, HIGH);
        delay(1000);
    
  }
  else if (r_p==3){
    
        digitalWrite(LED_BUILTIN, LOW);
        delay(1500);
        digitalWrite(LED_BUILTIN, HIGH);
        delay(1500);
    } */
}


//motion using ir sensor
void motion(int m_p){
  Serial.println(m_p);
  int val = 0;
  val = digitalRead(D8);
  if (val == HIGH){
    digitalWrite(D4, HIGH);
    Serial.println(m_p); 
  }
  else {
    digitalWrite(D4, LOW);

}
}
