import os,sys
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_models(X_train,X_test,y_train,y_test,models):
    '''
    This function evaluates the accuracy and efficiency of the models given 
    '''
    report=dict()
    for key,val in models.items():
        model=val
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)

        score=r2_score(y_test,y_pred)
        final={key:score}
        report.update(final)

    return report

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)



    