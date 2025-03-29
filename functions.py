from dashscope import Application
import json
import os
import requests
from alibabacloud_bailian20231229.client import Client
from alibabacloud_bailian20231229.models import ApplyFileUploadLeaseRequest, AddFileRequest
from alibabacloud_tea_util import models as util_models
import hashlib
from my_personal_key import workspace_id, category_id, config,my_personal_api_key,my_personal_app_id

client = Client(config)

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_pre_signed_url(file_path):
    file_md5 = calculate_md5(file_path)
    file_name = os.path.basename(file_path)
    request = ApplyFileUploadLeaseRequest(
        file_name=file_name,  # 替换为你的文件名
        md_5=str(file_md5),  # 替换为你的文件MD5值
        size_in_bytes=str(os.path.getsize(file_path)),  # 替换为你的文件大小（字节）
    )
    headers = {}
    runtime = util_models.RuntimeOptions()
    response = client.apply_file_upload_lease_with_options(category_id, workspace_id, request, headers, runtime)
    print(response.body)
    return response


def upload_file(pre_signed_url, file_path, headers):
    try:
        with open(file_path, 'rb') as file:
            response = requests.put(pre_signed_url, data=file, headers=headers)

        if response.status_code == 200:
            print("File uploaded successfully.")
        else:
            print(f"Failed to upload the file. ResponseCode: {response.status_code}")
        return response
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# 请求参数
def add_file(lease_id, headers):
    request = AddFileRequest(
        lease_id=lease_id,
        category_id=category_id,  # 替换为你的CategoryId
        category_type="UNSTRUCTURED",
        parser="DASHSCOPE_DOCMIND"

    )
    # 调用接口
    runtime = util_models.RuntimeOptions()
    response = client.add_file_with_options(workspace_id, request, headers, runtime)
    print("文档添加成功")
    print(response.body)
    return response

def get_file_id(file_path):
    response = get_pre_signed_url(file_path)
    lease_id = response.body.data.file_upload_lease_id
    pre_signed_url = response.body.data.param.url
    X_bailian_extra = response.body.data.param.headers['X-bailian-extra']
    Content_Type = response.body.data.param.headers['Content-Type']
    headers = {
        "X-bailian-extra": X_bailian_extra,
        "Content-Type": Content_Type
    }
    response2 = upload_file(pre_signed_url, file_path,headers)
    headers2 =response2.headers
    headers2['X-bailian-extra'] = X_bailian_extra
    headers2['Content-Type'] = Content_Type
    response3 =add_file(lease_id,headers2)
    return response3.body.data.file_id

def request_dashscope_api(message,file_id="file_f1c68368c4b54f6bb60eb166369233e3_10901437"):
    prompt = """
# 角色
你是一位道路交通情况描述大师，能够通过用户提供的资料和图片进行分析，并生成详细的道路交通情况报告。你的报告将涵盖交通流量分析、交通拥堵情况、交通事故情况以及交通设施情况。

## 技能
### 技能 1: 交通流量分析
- 根据用户提供的资料和图片，分析当前道路上的车辆数量和流动情况。

### 技能 2: 交通拥堵情况分析
- 识别道路是否拥堵,但是不要说高峰时段相关的内容。
- 如果图中有具体流量的统计数据，则需要根据提及该数据。
- 例如：up 35对应着右车道流量,down 36对应左侧车道流量。 

### 技能 3: 交通事故情况分析
- 识别是否有交通事故发生。
- 例如：就视频中所示，并没有发现交通事故。


## 约束
- 所有报告必须基于用户提供的资料和图片进行分析。
- 报告内容应全面覆盖交通流量分析、交通拥堵情况、交通事故情况。
- 报告措辞需专业且准确，避免使用模糊或不确定的表述。
- 默认使用英文撰写报告。
- 如果用户没有给出这个图片的时间，则不要提及时间。

{}
    """
    response = Application.call(
        app_id=my_personal_app_id,
        api_key=my_personal_api_key,
        prompt=prompt.format(message),
        rag_options={
            "session_file_ids": [file_id],  # FILE_ID1 替换为实际的临时文件ID,逗号隔开多个
        }
    )
    return response


def get_report(message,file_path):
    file_id = get_file_id(file_path)
    report = request_dashscope_api(message,file_id)
    report = report["output"]["text"]
    with open("road_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    return report






if __name__ == "__main__":
    # 替换为你的业务空间 ID 和类目 ID
    file_path = "./Dataset/Vehicle_Detection_Image_Dataset/train/images/1_mp4-1_jpg.rf.9115af93de4cbf89e48a9c33dfe996c8.jpg"
    message = "目前进入左侧道路经过车辆30，右侧道路经过车辆20，请您分析道路交通情况并给出建议。"
    result = get_report(message,file_path)
    print(result)

