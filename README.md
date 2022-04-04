# Three facial traits (Trustworthness, Dominance, Attractiveness) prediction flow
The picture of CEO contains lots of information. When seeing photo, we make some judgement about the person in it. In order to quantify what people see and feel. We conduct a model to train and predict the htree facial traits , Trustworthness, Dominance and Attractiveness of a CEO. In these article, the steps of how we predict those traits are not shown here. Only offer the trained model by python. The main target is to show how to use it to get the three traits of the photo user collected. First, we have to say there is a constraint, that the number of photos should be greater than 30 when using these code and modes. The demo data and necessary file to use is in dataset.

## Data and Photo
A data set of list of CEO is required in the begining, here is the example of excel file that must contain these column

![image](https://user-images.githubusercontent.com/37869717/161537080-07451670-b0d9-4124-a2e2-2eba00ab09f5.png)

"TICKER" is the ticker of a company, "GLASS","BEARD","STUBBLE" are indicator of whether the person in picture is wearing a glasses, have beard and have stubble. These should be manually conduct when collecting photo, these columns are what will be used in next step of stages.

## Python code 
### preamble
After collecting data and noted those three columns, we can flip to next part, which is python code. In this article, we use a lot of modules and packages, The  preamble.ipynb contains code of those modules. If user have not downloaded before, please take some time to finishing downlaoding or updateing those modules and packages first.

### functions of calculating of photo
In this part, we define lots of functions, some of it is refer some code from others, but I can't remember where I found these. If user know the source, please tell me, that I could add the reference of those code written by others.

1.1 visualize andmark

```python
def visualize_landmark(image_array, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    for facial_feature in landmarks.keys():
        draw.point(landmarks[facial_feature],fill='#28FF28')
    imshow(origin_img)
```                 

1.2 point center of two eyes


```python
def align_face(image_array, landmarks):
    """ align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # get list landmarks of left and right eye
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    # calculate the mean point of landmarks of left and right eye
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    # at the eye_center, rotate the image by the angle
    rotate_matrix =cv2.getRotationMatrix2D((int(eye_center[0]),int(eye_center[1])), angle, scale=1)
    #center must be integer
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle
``` 
1.3 rotate picture

```python
def rotate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)
```

1.4 rotate landmarks

```python
def rotate_landmarks(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    rotated_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            rotated_landmark = rotate(origin=eye_center, point=landmark, angle=angle, row=row)
            rotated_landmarks[facial_feature].append(rotated_landmark)
    return rotated_landmarks
``` 

1.5 corp picture

```python
def corp_face(image_array, landmarks):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    """

    eye_landmark = np.concatenate([np.array(landmarks['left_eye']),
                                   np.array(landmarks['right_eye'])])
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    lip_landmark = np.concatenate([np.array(landmarks['top_lip']),
                                   np.array(landmarks['bottom_lip'])])
    lip_center = np.mean(lip_landmark, axis=0).astype("int")
    mid_part = lip_center[1] - eye_center[1]
    top = 1.2*(eye_center[1] - mid_part * 30 / 15)#35
    bottom = 1.1*(lip_center[1] + mid_part )

    w = h = bottom - top
    x_min = np.min(landmarks['chin'], axis=0)[0]
    x_max = np.max(landmarks['chin'], axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - w / 2, x_center + w / 2)

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    return cropped_img,left, top
``` 


1.6 chop background

```python
def chop_back(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_land)
    rect = detector(img)[0]
    sp = predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])
    outline = landmarks[[*range(17), *range(26,16,-1)]]
    Y, X = skimage.draw.polygon(outline[:,1], outline[:,0])
    cropped_img = np.ones(img.shape, dtype=np.uint8)*255
    cropped_img[Y, X] = img[Y, X]
    return Image.fromarray(cropped_img)
``` 

1.7 polygon area,centeroidnp, distance ,x^2 coefficient calucation

```python
#多變形面積計算
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

#質心計算
def centeroidnp(x):
    arr = landmarks[x]
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

#距離(v_h:0=horizontal, 1=vertical)
def distance_ca(y1,y2,v_h):
    x1 = landmarks[y1]
    x2 = landmarks[y2]
    length1 = x1.shape[0]
    length2 = x2.shape[0]
    sum1 = np.sum(x1[:, v_h])
    sum2 = np.sum(x2[:, v_h])
    dis = abs(sum1/length1-sum2/length2)
    return dis

#計算回歸係數
def grad_ca(arr):
    x = np.array(list(zip(*landmarks[arr]))[0]).reshape([-1, 1])
    y = np.array(list(zip(*landmarks[arr]))[1])
    model = LinearRegression().fit(x, y) # sklearn.linear_model
    return model.coef_[0]

#計算x^2回歸係數
def gradx2_ca(arr):
    x = np.array(list(zip(*landmarks[arr]))[0]).reshape([-1, 1])
    y = np.array(list(zip(*landmarks[arr]))[1])
    x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
    model = LinearRegression().fit(x_, y) # sklearn.linear_model
    return model.coef_[1]
``` 

1.8 HSV calucation

```python
import colorsys
import PIL.Image as Image
 
def get_dominant_color(image):
    max_score = 0.0001
    dominant_color = None
    for count,(r,g,b) in image.getcolors(image.size[0]*image.size[1]):
        # 转为HSV标准
        saturation = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)[1]
        y = min(abs(r*2104+g*4130+b*802+4096+131072)>>13,235)
        y = (y-16.0)/(235-16)
    #忽略高亮色
        if y > 0.9:
            continue
        score = (saturation+0.1)*count
        if score > max_score:
            max_score = score
            dominant_color = [r,g,b]
    return colorsys.rgb_to_hsv(dominant_color[0],dominant_color[1],dominant_color[2])


def chop_area(img, outline):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_land)
    rect = detector(img)[0]
    sp = predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])
    #outline = landmarks[[*range(17), *range(26,16,-1)]]
    Y, X = skimage.draw.polygon(outline[:,1], outline[:,0])
    cropped_img = np.ones(aaa.shape, dtype=np.uint8)*255
    cropped_img[Y, X] = img[Y, X]
    return Image.fromarray(cropped_img)
``` 

1.9 entropy color calucation

```python
def get_entropy_color(image):
    max_score = 0.0001
    dominant_color = None
    h_dist=[]
    s_dist=[]
    v_dist=[]
    for count,(r,g,b) in image.getcolors(image.size[0]*image.size[1]):
        # GRB to HSV
        saturation = colorsys.rgb_to_hsv(r, g, b)
    #忽略高亮色
        if r+g+b < 750:
            h_dist=h_dist+[saturation[0]]
            s_dist=s_dist+[saturation[1]]
            v_dist=v_dist+[saturation[2]]
    return h_dist,s_dist,v_dist


def hist_entropy(h):
    hist_h = np.histogram(h, bins=50)
    hist_h = list(hist_h[0])
    hist_h = [h/float(sum(hist_h)) for h in hist_h]
    entropy_h = entropy(hist_h)
    return entropy_h
```


1.10 excel output function

```python
def append_df_to_excel(filename, df, sheet_name='Sheet1', startrow=None,
                       truncate_sheet=False, 
                       **to_excel_kwargs):
    if not os.path.isfile(filename):
        df.to_excel(
            filename,
            sheet_name=sheet_name, 
            startrow=startrow if startrow is not None else 0, 
            **to_excel_kwargs)
        return
    if 'engine' in to_excel_kwargs:
        to_excel_kwargs.pop('engine')
    writer = pd.ExcelWriter(filename, engine='openpyxl', mode='a')
    writer.book = load_workbook(filename)
    if startrow is None and sheet_name in writer.book.sheetnames:
        startrow = writer.book[sheet_name].max_row
    if truncate_sheet and sheet_name in writer.book.sheetnames:
        idx = writer.book.sheetnames.index(sheet_name)
        writer.book.remove(writer.book.worksheets[idx])
        writer.book.create_sheet(sheet_name, idx)
    writer.sheets = {ws.title:ws for ws in writer.book.worksheets}
    if startrow is None:
        startrow = 0
    df.to_excel(writer, sheet_name, startrow=startrow, **to_excel_kwargs)
    writer.save()
    
def my_mode(sample):
    c = Counter(sample)
    return [k for k, v in c.items() if v == c.most_common(1)[0][1]]
``` 

1.11 aggregate whole function

- read photo folder
- read photo
- photo manipulate
- 68 landmarks calucation
- 65 attribute and FWHR caculation 
- devide area and length related attribute by heqad area(att_01) 
- standerdize whole set of data
- import three traits prediction model
- out put data

```python
def attr65_cal_fn(file):
    #讀取路徑照片
    path = os.getcwd()+file
    extensions = [r'.jpg', '.png', '.jpeg','.jfif']
    df = pd.DataFrame()
    global images
    images = [x for x in Path(path).iterdir() if x.suffix.lower() in extensions]
    indx = 0
    #尋找68點轉換成65臉部結構變數，並計算長寬比
    for i in images:
        #產生需要的index
        img_id = str(i).split('\\')[-1][:-4].strip('.')    
        img_id1 = img_id.split('_')[0]
        year  = img_id.split('_')[1]
        print(indx,':',img_id)
        indx = indx+1
        #IMAGE RESIZE AND ROTATE
        image_array = cv2.imread(str(i))
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        face_landmarks_list = face_recognition.face_landmarks(image_array, model="large")
        face_landmarks_dict = face_landmarks_list[0]
        aligned_face, eye_center, angle = align_face(image_array=image_array, landmarks=face_landmarks_dict)
        rotated_landmarks = rotate_landmarks(landmarks=face_landmarks_dict,
                                         eye_center=eye_center, angle=angle, row=image_array.shape[0])
        cropped_img,left, top= corp_face(image_array=aligned_face, landmarks=rotated_landmarks)
        global aaa
        aaa = cv2.resize(cropped_img, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(path_land)
        rect = detector(aaa)[0]
        sp = predictor(aaa, rect)
        global landmarks
        #最終的68點輸出
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        #計算65臉部特徵點
        att_01 = PolyArea(*zip(*landmarks[0:18],*landmarks[27:18]))
        att_02 =distance_ca(range(19,25),range(7,10),1)
        att_03 = distance_ca(range(15,17),range(0,2),0)
        tmp = centeroidnp(range(27,37))
        att_04 = tmp[0]
        att_05 = tmp[1]
        att_06 =0
        att_07 =PolyArea(*zip(*landmarks[17:22]))+ PolyArea(*zip(*landmarks[22:27]))
        att_08 = distance_ca([17,21,22,26],[18,19,20,23,24,25],1)
        att_09 = distance_ca([21], [26], 0)+distance_ca([17], [22], 0)
        att_10 = grad_ca(range(19,22))
        att_11 = PolyArea(*zip(*landmarks[36:42]))+ PolyArea(*zip(*landmarks[42:48]))
        att_12 = PolyArea(*zip(*landmarks[37:42]))+ PolyArea(*zip(*landmarks[43:48]))
        att_13 = distance_ca([40,41,46,47],[37,38,43,44],1)
        att_14 = distance_ca([39,45],[36,42],0)
        att_15 = att_12*1/np.pi/att_13**2
        att_16 = PolyArea(*zip(*landmarks[30:36]))+PolyArea(*zip(*landmarks[27:32]))+PolyArea(*zip(*landmarks[[27,30,35]]))
        att_17 = distance_ca([33],[27],1)
        att_18 = distance_ca([35],[31],0) 
        att_19 = gradx2_ca(range(31,36))
        att_20 = abs((np.array(centeroidnp([34,32]))-np.array(centeroidnp([31,35]))))[1] 
        att_21 = abs((np.array(centeroidnp([7,9]))-np.array(centeroidnp([2,14]))))[1]
        att_22 = grad_ca([6,7,8])+abs(grad_ca([8,9,10]))
        mjar   = np.mean(np.array(list(zip(*landmarks[2:15]))),1)
        att_23 = np.std(np.sqrt(np.sum((landmarks[2:15]-mjar)**2,1)))
        att_24 = abs(gradx2_ca(range(6,11)))
        att_25 = PolyArea(*zip(*landmarks[48:60]))
        att_26 = abs((np.array(centeroidnp(range(48,55)))-np.array(centeroidnp([55,56,57,58,59,48,54]))))[1]
        att_27 = abs((np.array(centeroidnp(range(48,55)))-np.array(centeroidnp([60, 61, 62, 63, 64, 48, 54]))))[1]
        att_28 = abs((np.array(centeroidnp([65,66,67,48,54]))-np.array(centeroidnp([55,56,57,58,59, 48, 54]))))[1]
        att_29 = abs((np.array(centeroidnp([48]))-np.array(centeroidnp([54]))))[0]
        att_30 = abs((np.array(centeroidnp([65,66,67,48,54]))-np.array(centeroidnp([60, 61, 62, 63, 48, 54]))))[0]
        att_31 = abs(gradx2_ca([60,61,62,63,48,54]))
        att_32 = abs(gradx2_ca([65,66,67,48,54]))
        att_33 = abs((np.array(centeroidnp([65,66,67,48,54]))-np.array(centeroidnp([60, 61, 62, 63, 48, 54]))))[0]
        att_34 = abs((np.array(centeroidnp([7,9]))-np.array(centeroidnp([2,3,31,48]))))[1]
        att_35 = abs(grad_ca([2,3,31,48]))
        tmp    = np.array(centeroidnp([27]))-np.array(centeroidnp([28,39]))
        att_36 = abs(tmp[1]/tmp[0])
        att_37 = abs((np.array(centeroidnp(range(7,10)))-np.array(centeroidnp(range(36,48)))))[1]/att_02
        att_38 = abs((np.array(centeroidnp(range(7,10)))-np.array(centeroidnp(range(17,26)))))[1]/att_02
        att_39 = abs((np.array(centeroidnp(range(7,10)))-np.array(centeroidnp(range(48,60)))))[1]/att_02
        att_40 = abs((np.array(centeroidnp(range(7,10)))-np.array(centeroidnp(range(27,36)))))[1]/att_02
        att_41 = abs((np.array(centeroidnp(range(37,42)))-np.array(centeroidnp(range(43,48)))))[0]
        att_42 = abs((np.array(centeroidnp([39,42]))-np.array(centeroidnp([50,52]))))[1]
        att_43 = abs((np.array(centeroidnp([17,21,22,26]))-np.array(centeroidnp([27,28,43,44]))))[1]
        att_44 = abs((np.array(centeroidnp([0,2]))-np.array(centeroidnp([36]))))[0]
        att_45 = abs((np.array(centeroidnp([14,16]))-np.array(centeroidnp([45]))))[0]
        att_46 = abs((np.array(centeroidnp([56,58]))-np.array(centeroidnp([7, 9]))))[1]
        att_47 = abs((np.array(centeroidnp([32,34]))-np.array(centeroidnp([50,51]))))[1]
        #att_48-att_62 = HSV calculation
        skin_area = chop_area(aaa,landmarks[[*range(17), *range(26,16,-1)]])
        image_skin = skin_area.convert("RGB")
        tmp = get_dominant_color(image_skin)
        att_48 = tmp[0]
        att_49 = tmp[1]
        att_50 = tmp[2]

        eyebrow_area = chop_area(aaa, landmarks[[*range(17,22,1), *range(22,27,1),22,21,17]])
        image_eyebrow = eyebrow_area.convert("RGB")
        tmp1 = get_dominant_color(image_eyebrow)

        att_51 = tmp1[0]
        att_52 = tmp1[1]
        att_53 = tmp1[2]

        lips_area = chop_area(aaa, landmarks[range(48,60)])
        image_lips = lips_area.convert("RGB")
        tmp2 = get_dominant_color(image_lips)

        att_54 = tmp2[0]
        att_55 = tmp2[1]
        att_56 = tmp2[2]

        eye_area = chop_area(aaa, landmarks[[40,41,37,38,40,47,43,44,46,47]])
        image_eye = eye_area.convert("RGB")
        tmp3 = get_dominant_color(image_eye)

        att_57 = tmp3[0]
        att_58 = tmp3[1]
        att_59 = tmp3[2]

        skin_area = chop_area(aaa,landmarks[[*range(17), *range(26,16,-1)]])
        skin_image = skin_area.convert("RGB")
        hsv_dist = get_entropy_color(skin_image)

        att_60 = hist_entropy(hsv_dist[0])
        att_61 = hist_entropy(hsv_dist[1])
        att_62 = hist_entropy(hsv_dist[2])

        #人工註計資料
        att_63=int(ceo_data[ceo_data['TICKER']==img_id1]['GLASS'])
        att_64=int(ceo_data[ceo_data['TICKER']==img_id1]['BEARD'])
        att_65=int(ceo_data[ceo_data['TICKER']==img_id1]['STUBBLE'])
        fwhr = distance_ca([17],[26],0)/distance_ca([41,46],range(50,52),1)
        #合併為DATAFRAME
        d = {'coName': [img_id1],'att_01': [att_01],'att_02': [att_02],'att_03': [att_03],'att_04': [att_04],'att_05': [att_05],
             'att_06': [att_06],'att_07': [att_07],'att_08': [att_08],'att_09': [att_09],'att_10': [att_10],'att_11': [att_11],
             'att_12': [att_12],'att_13': [att_13],'att_14': [att_14],'att_15': [att_15],'att_16': [att_16],'att_17': [att_17],
             'att_18': [att_18],'att_19': [att_19],'att_20': [att_20],'att_21': [att_21],'att_22': [att_22],'att_23': [att_23],
             'att_24': [att_24],'att_25': [att_25],'att_26': [att_26],'att_27': [att_27],'att_28': [att_28],'att_29': [att_29],
             'att_30': [att_30],'att_31': [att_31],'att_32': [att_32],'att_33': [att_33],'att_34': [att_34],'att_35': [att_35],
             'att_36': [att_36],'att_37': [att_37],'att_38': [att_38],'att_39': [att_39],'att_40': [att_40],'att_41': [att_41],
             'att_42': [att_42],'att_43': [att_43],'att_44': [att_44],'att_45': [att_45],'att_46': [att_46],'att_47': [att_47],
             'att_48': [att_48],'att_49': [att_49],'att_50': [att_50],'att_51': [att_51],'att_52': [att_52],'att_53': [att_53],
             'att_54': [att_54],'att_55': [att_55],'att_56': [att_56],'att_57': [att_57],'att_58': [att_58],'att_59': [att_59],
             'att_60': [att_60],'att_61': [att_61],'att_62': [att_62],'att_63': [att_63],'att_64': [att_64],'att_65': [att_65],'year':[year],'FWHR': [fwhr]}
        data = pd.DataFrame(data=d)
        df = df.append(data)
    #65運算完後將面積或長度變數除以臉部面積    
    to_scale_list = [2,3,4,5,7,8,9,11,12,13,14,15,16,17,18,19,21,22,25,26,27,28,29,30,33,41,42,43,44,45,46,47]
    df.iloc[:,to_scale_list] = df.iloc[:,to_scale_list]/np.array(df.iloc[:,[1]])
    #標準化，因模型本身使用標準化後數據
    sc = StandardScaler()
    df_d = df.iloc[:, 1:66]
    XX = sc.fit_transform(df_d)
    XX = sm.add_constant(df_d, prepend=True)
    scaler_x = MinMaxScaler()
    scaler_x.fit(XX)
    xtrain_scale=scaler_x.transform(XX)
    #套用已建立模型
    res = pd.DataFrame(data={'Trust':list(itertools.chain.from_iterable(model_trust.predict(xtrain_scale).tolist())),
                             'Dom':list(itertools.chain.from_iterable(model_dom.predict(xtrain_scale).tolist())),
                             'Attr':list(itertools.chain.from_iterable(model_attr.predict(xtrain_scale).tolist()))
                            })   
    df.reset_index(drop=True, inplace=True)
    res.reset_index(drop=True, inplace=True)
    df_RES = pd.concat([df, res], axis=1)
    #輸出結果路徑
    append_df_to_excel(os.getcwd()+r'\analysis_data.xlsx',df_RES, sheet_name='ceo_3traits', index = None)
    return  res
``` 
## execute to get final result
```python
model_trust = load_model(os.getcwd()+r'\p_trust.h5')
model_dom = load_model(os.getcwd()+r'\p_dom.h5')
model_attr = load_model(os.getcwd()+r'\p_attr.h5')
ceo_data = pd.read_excel(os.getcwd()+r'\DEMO.xlsx', sheet_name=0)
files = r"\DEMO_PHOTO"
x = attr65_cal_fn(files)
``` 
