from src.database.config import supabase
import bcrypt

def hash_password(pwd):
    return bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()

def check_pwd(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def check_teacher_exists(username):
    response = supabase.table("teachers").select("username").eq("username", username).execute()
    return len(response.data)>0;

def create_teacher(username, name, password):
    user = {"username" : username, "name" : name, "password" : hash_password(password)}
    response = supabase.table("teachers").insert(user).execute()
    return response.data

def teacher_login(username, password):
    response = supabase.table("teachers").select("*").eq("username", username).execute()
    if(response.data):
        teacher = response.data[0]
        if(check_pwd(password, teacher['password'])):
            return teacher
        return None