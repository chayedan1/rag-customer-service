import os
import shutil
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# 1. Compress directory to zip
print("Zipping KownledgeBase/手册/插图...")
# shutil.make_archive takes base_name, format, root_dir
# It will create KownledgeBase/手册/插图.zip from KownledgeBase/手册/插图/
zip_base = os.path.join("KownledgeBase", "手册", "插图")
zip_root = os.path.join("KownledgeBase", "手册", "插图")
if os.path.exists(zip_root):
    shutil.make_archive(zip_base, "zip", zip_root)
    print("Zipped successfully!")
else:
    print("Error: KownledgeBase/手册/插图/ not found!")

# 2. Re-create .gitignore in UTF-8
print("Writing .gitignore...")
with open(".gitignore", "w", encoding="utf-8", newline="\n") as f:
    f.write("__pycache__/\n*.pyc\nKownledgeBase/手册/插图/\nprepare_repo.py\n")

# 3. Clean and Re-initialize git
print("Cleaning old .git...")
# On Windows, rmtree might fail if files are read-only, so we run a system command or try multiple times
def force_delete_git():
    if os.path.exists(".git"):
        subprocess.run(["attrib", "-h", "-r", "-s", ".git/*", "/s", "/d"], shell=True)
        shutil.rmtree(".git", ignore_errors=True)
        if os.path.exists(".git"):
            subprocess.run("rmdir /s /q .git", shell=True)

force_delete_git()

# Run git commands
print("Re-initializing Git...")
subprocess.run(["git", "init"], check=True)
subprocess.run(["git", "lfs", "install"], check=True)
subprocess.run(["git", "lfs", "track", "vector_index.pkl"], check=True)
subprocess.run(["git", "lfs", "track", "KownledgeBase/手册/插图.zip"], check=True)

# Add files explicitly to avoid adding unzipped images
print("Adding files to git index...")
files_to_add = [
    ".gitattributes",
    ".gitignore",
    "Dockerfile",
    "app.py",
    "graph.py",
    "requirements.txt",
    "vector_index.pkl",
    "vector_store.py",
    os.path.join("KownledgeBase", "手册", "插图.zip")
]

# Add all txt manual files in KownledgeBase/手册/
manuals_dir = os.path.join("KownledgeBase", "手册")
for file in os.listdir(manuals_dir):
    if file.endswith(".txt"):
        files_to_add.append(os.path.join(manuals_dir, file))

for file_path in files_to_add:
    if os.path.exists(file_path):
        subprocess.run(["git", "add", file_path], check=True)

subprocess.run(["git", "commit", "-m", "deploy: clean init with LFS zip file"], check=True)
subprocess.run(["git", "branch", "-M", "main"], check=True)
subprocess.run(["git", "remote", "add", "origin", "http://oauth2:ms-e183fd2d-28b1-4e02-b65e-b15ecf3541e7@www.modelscope.cn/studios/chayedan123/rag-customer-service.git"], check=True)

print("=== ALL STEPS COMPLETED SUCCESSFULLY ===")
