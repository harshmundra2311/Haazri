import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st
from src.database.db import get_all_students


@st.cache_resource
def load_dlib_models():
    detector=dlib.get_frontal_face_detector()
    
    sp=dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )
    
    facerec=dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )
    
    return detector, sp, facerec

def get_face_embeddings(image_np):
    detector,sp,facerec=load_dlib_models()
    faces=detector(image_np,1)
    
    encodings=[]
    
    for face in faces:
        shape=sp(image_np,face)
        face_descriptor=facerec.compute_face_descriptor(image_np,shape,1)#128 embedding
        
        encodings.append(np.array(face_descriptor))
    return encodings


@st.cache_resource
def get_trained_model():
    X=[]
    y=[]
    
    student_db=get_all_students()
    
    if not student_db:
        return None
    
    for student in student_db:
        embedding=student.get('face_embedding')
        if embedding:
            X.append(np.array(embedding))
            y.append(student.get('student_id'))
            
    if len(X)==0:
        return 0
    
    clf=SVC(kernel='linear',probability=True,class_weight='balanced')
    
    try:
        clf.fit(X,y)
    except ValueError:
        pass
    
    return {'clf':clf,'X':X,"y":y}
    
def train_classifier():
    st.cache_resource.clear()
    model_data=get_trained_model()
    return bool(model_data)

def predict_attendance(class_image_np):
    encodings = get_face_embeddings(class_image_np)

    detected_student = {}

    model_data = get_trained_model()

    if not model_data:
        return detected_student, [], len(encodings)

    X_train = model_data['X']
    y_train = model_data['y']

    all_students = sorted(list(set(y_train)))

    resemblance_threshold = 0.42
    min_margin = 0.08

    for encoding in encodings:

        distances = [np.linalg.norm(x - encoding) for x in X_train]

        sorted_idx = np.argsort(distances)
        best_idx = int(sorted_idx[0])
        best_match_score = distances[best_idx]
        predicted_id = int(y_train[best_idx])

        is_confident = True
        if len(sorted_idx) > 1:
            for idx in sorted_idx[1:]:
                if y_train[idx] != predicted_id:
                    runner_up_score = distances[idx]
                    is_confident = (runner_up_score - best_match_score) >= min_margin
                    break

        if best_match_score <= resemblance_threshold and is_confident:
            detected_student[predicted_id] = True

    return detected_student, all_students, len(encodings)