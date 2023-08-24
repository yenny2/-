import argparse # 파이썬 명령행 파싱 모듈
import io #
import os
import json
import glob #
from PIL import Image
from uuid import uuid4 # 유니버설 고유 식별자 생성을 위한 모듈.
import torch
from flask import Flask, render_template, request, redirect, url_for, session
from ast import literal_eval # 문자열을 파이썬 자료구조로 변환
import collections # 컨테이너에 저장된 요소들을 세는데 사용되는 모듈
import numpy as np
import sys
import pandas as pd
from flask_session import Session
from ultralytics import YOLO
import cv2
clsa={0: 'broken_line', 1: 'circle', 2: 'diagonal_line', 3: 'dot', 
      4: 'out_line', 5: 'quarte_circle', 6: 'scratch', 
      7: 'semicircle', 8: 'short_line', 9: 'aquare_line',
      10: 'thick_line', 11: 'thin_line'}

# 폴더 안의 파일 삭제 함수
def deleteallfiles(filepath):
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            os.remove(file.path)
            
app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
# 웹 서버에서 이미지를 업로드하고 yolo 모델을 사용하여 이미지에서 객체를 탐지한 결과를 반환
@app.route("/detect", methods=["GET","POST"])
def detect():
    global clsa
    if request.method == "POST":
        deleteallfiles('static/aft')
        deleteallfiles('static/bef')
        # 다중파일 업로드
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files.getlist("file")
        if not file:
            return
        
        results_list = [] 
        resultlist = []   #결과 리스트
        pf = []   #pass fail 리스트 
        inum=0
        for file in file:
            inum+=1
            filename = file.filename.rsplit("/")[0]   #파일 경로에서 파일명만 추출
            print("진행 중 파일 :", filename)
            
            img_bytes = file.read()
            #바이트스트림: 파일과 유사한 인터페이스를 가진 객체
            img = Image.open(io.BytesIO(img_bytes))   #img_bytes를 바이트 스트림으로 변환
            #img 객체를 static/bef/ 디렉토리에 filename 이름으로 jpeg포멧의 이미지 파일로 저장
            img.save(f"static/bef/{inum}{filename}", format="JPEG")
            
            print('원본 저장')
            
            model = YOLO('C:/hmkd1/web/vision_flask/bestm.pt')
            results = model(img)
            print("보기1",results)
            for i in range(len(results)):
                img1=results[i].plot()
                cv2.imwrite((f'static/aft/{inum}{filename}.JPEG'), img1)
                
                #탐지된 객체의 이름을 가져옴
                data = results[i].boxes.cls
                data_counter = collections.Counter(data.numpy())
                print(data_counter)
                print('데이터', data)
                
                if len(data) == 0:
                    pf.append('PASS')
                if len(data) != 0:
                    pf.append('FAIL')
                
                li=[]
                for d in range(len(data)):
                    num=int(data[d])
                    if num in clsa:
                       li.append(clsa[num])
                       li=list(set(li))
                resultlist.append(li)       
                
            root = 'static/aft'
            if not os.path.isdir(root):
                return 'Error: not found!'
            files = []
            for file in glob.glob('{}/*.*'.format(root)):
                fname = file.split(os.sep)[-1]
                files.append(fname)
            print("파일스:", files)
            
            if len(files) > 0:
                firstimage = 'static/aft/' + files[0]
            else:
                pass
            
            datanum = len(pf)
            rate = round(pf.count('PASS') / len(pf), 3)
            correct = pf.count('PASS')
            
            print("리스트 내용: ",resultlist)
            result_df = pd.DataFrame({
            'File': files,
            'Result': resultlist
            })
            result_df["정상 유무"]=0
            for i in range(len(result_df)):
                print("내용",i,'  ',result_df.Result[i])
                if result_df.Result[i]==[]:
                    result_df["정상 유무"][i]="정상"
                else:
                    result_df["정상 유무"][i]="불량"
        print(result_df)
        session['result_df'] = result_df
                
        return render_template("imageshow.html", files=files, resultlist=resultlist, pf=pf,datanum=datanum,
                               rate=rate, correct=correct,firstimage=firstimage,enumerate=enumerate,len=len,
                               results_list=results_list)
    return render_template('detect.html')

# @app.route('/suyul')
# def suyul():
#     result=session.get('result_df')
#     countdf=result["정상 유무"].value_counts()
#     if countdf.count()==1:
#         suyul=0
#     else:
#         truenum=countdf["정상"]
#         falsenum=countdf["불량"]
#         suyul=(truenum/(truenum+falsenum))*100
#     print(suyul)
#     return render_template('suyul.html',tables=[result.to_html()],titles=result.columns.values,suyul=round(suyul,2))

# @app.route('/dataflow')
# def dataflow():
#     return render_template('dataflow.html')

# 기존 페이지
@app.route('/')
def home():
    return render_template('index.html')
            
            
if __name__=='__main__':
    parser = argparse.ArgumentParser(
    description = 'Flask app exposing yolov8 models')
    parser.add_argument('--port', default=5000, type=int, help='port number')
    args = parser.parse_args()
    
    #local model
    
    # model = YOLO('C:/Users/human/Desktop/학원/flask/v35n.pt')
    # model.eval() 
    
    flask_options = dict(
        host = '0.0.0.0',
        debug = True,
        port = args.port,
        threaded = True,
    )
        
    app.run(**flask_options)