import gradio as gr
import joblib
import pandas as pd
import spaces 

#1- Modeli uygulamada 1 kez yüklüyecek

model = joblib.load('kredi_modeli.joblib')

#2- ZEROGPU donanımı - 
@spaces.GPU(duration=1)
def zerogpu_uyumluluk():
  return "kullanılmıyor"

#3- Tahmin Fonksiyonu 

def kredi_tahmin(yas, calisma_tipi, calisma_suresi_yil,ev_durumu, aylik_gelir,kredi_skoru,kredi_tutari,vade_ay):
  basvuru = pd.DataFrame([{
      "yas": yas,
      "calisma_tipi": calisma_tipi,
      "calisma_suresi_yil": calisma_suresi_yil,
      "ev_durumu": ev_durumu,
      "aylik_gelir": aylik_gelir,
      "kredi_skoru": kredi_skoru,
      "kredi_tutari": kredi_tutari,
      "vade_ay": vade_ay
  }])

  olasiliklar = model.predict_proba(basvuru)[0]
  return {"Kredin Reddedilir": olasiliklar[0],
          "Kredin Kabul Edilir": olasiliklar[1]}

#4- Arayüz Bileşenleri

girdiler = [
    gr.Slider(18,65,value=34,step=1,label="Yaş"),
    gr.Dropdown(["Özel Sektör","Kamu","Serbest"],label="Çalışma Tipi",value="Özel Sektör"),
    gr.Slider(0,60,value=8,step=1,label="Çalışma Süresi"),
    gr.Radio(["Kira","Ev Sahibi","Aile Yanı"],label="Ev Durumu",value="Kira"),
    gr.Number(value=65000,label="Aylık Gelir"),
    gr.Slider(0,3000,value=1450,step=20,label="Kredi Skoru"),
    gr.Number(value=250000,label="Kredi Tutarı"),
    gr.Slider(12,48,value=36,step=1,label="Vade Ayı")
]

cikti = gr.Label(num_top_classes=2 ,label="Tahmin Sonucu")

demo = gr.Interface(
    fn=kredi_tahmin,
    inputs=girdiler,
    outputs=cikti,
    title="Kredi Onay Tahmini",
    description="ECS BANK - Kredi Tahmin Uygulaması",
    flagging_mode="never",
)

#5- Uygulamayı Başlat
demo.launch()
