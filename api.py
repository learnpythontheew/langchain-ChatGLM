import argparse
import json
import os
import shutil
from typing import List, Optional
import urllib
import asyncio
import nltk
import pydantic
import uvicorn
from fastapi import Body, FastAPI, File, Form, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing_extensions import Annotated
from starlette.responses import RedirectResponse

from chains.local_doc_qa import LocalDocQA
from configs.model_config import (KB_ROOT_PATH, EMBEDDING_DEVICE,
                                  EMBEDDING_MODEL, NLTK_DATA_PATH,
                                  VECTOR_SEARCH_TOP_K, LLM_HISTORY_LEN, OPEN_CROSS_DOMAIN)
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint
# from langchain.prompts import PromptTemplate
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
# from datetime import datetime
import uuid
import logging
# from jsonformer import Jsonformer

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

from langchain.output_parsers import PydanticOutputParser

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


# 启动模型和fastapi

global app
global local_doc_qa
app = FastAPI()

parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--reload",type=bool,default=True)
# parser.add_argument("--timeout-keep-alive",type=int,default=5000)
# # 初始化消息
args = None
args = parser.parse_args()
args_dict = vars(args)
shared.loaderCheckPoint = LoaderCheckPoint(args_dict)

llm_model_ins = shared.loaderLLM()
llm_model_ins.set_history_len(LLM_HISTORY_LEN)

# # app = FastAPI()
# # Add CORS middleware to allow all origins
# # 在config.py中设置OPEN_DOMAIN=True，允许跨域
# set OPEN_DOMAIN=True in config.py to allow cross-domain
if OPEN_CROSS_DOMAIN:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


local_doc_qa = LocalDocQA()
local_doc_qa.init_cfg(
    llm_model=llm_model_ins,
    embedding_model=EMBEDDING_MODEL,
    embedding_device=EMBEDDING_DEVICE,
    top_k=VECTOR_SEARCH_TOP_K,
)


# response model

class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="HTTP status code")
    msg: str = pydantic.Field("success", description="HTTP status message")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class ListDocsResponse(BaseResponse):
    data: List[str] = pydantic.Field(..., description="List of document names")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }


class ChatMessage(BaseModel):
    question: str = pydantic.Field(..., description="Question text")
    response: str = pydantic.Field(..., description="Response text")
    history: List[List[str]] = pydantic.Field(..., description="History text")
    source_documents: List[str] = pydantic.Field(
        ..., description="List of source documents and their scores"
    )

    class Config:
        schema_extra = {
            "example": {
                "question": "工伤保险如何办理？",
                "response": "根据已知信息，可以总结如下：\n\n1. 参保单位为员工缴纳工伤保险费，以保障员工在发生工伤时能够获得相应的待遇。\n2. 不同地区的工伤保险缴费规定可能有所不同，需要向当地社保部门咨询以了解具体的缴费标准和规定。\n3. 工伤从业人员及其近亲属需要申请工伤认定，确认享受的待遇资格，并按时缴纳工伤保险费。\n4. 工伤保险待遇包括工伤医疗、康复、辅助器具配置费用、伤残待遇、工亡待遇、一次性工亡补助金等。\n5. 工伤保险待遇领取资格认证包括长期待遇领取人员认证和一次性待遇领取人员认证。\n6. 工伤保险基金支付的待遇项目包括工伤医疗待遇、康复待遇、辅助器具配置费用、一次性工亡补助金、丧葬补助金等。",
                "history": [
                    [
                        "工伤保险是什么？",
                        "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                    ]
                ],
                "source_documents": [
                    "出处 [1] 广州市单位从业的特定人员参加工伤保险办事指引.docx：\n\n\t( 一)  从业单位  (组织)  按“自愿参保”原则，  为未建 立劳动关系的特定从业人员单项参加工伤保险 、缴纳工伤保 险费。",
                    "出处 [2] ...",
                    "出处 [3] ...",
                ],
            }
        }



class WorkExperience(BaseModel):
    work_start_time: str = pydantic.Field(..., title="工作开始时间", description="请输入 YYYY-MM-DD 格式的日期")
    work_end_time: str = pydantic.Field(..., title="工作结束时间", description="请输入 YYYY-MM-DD 格式的日期")
    company: str = pydantic.Field(..., title="公司", description="请输入公司全称")
    industry: str = pydantic.Field(..., title="行业", description="请输入所属行业")
    position: str = pydantic.Field(..., title="职位", description="请输入职位名称")
    work_content: str = pydantic.Field(..., title="工作内容", description="请输入工作内容简介")

class EducationExperience(BaseModel):
    study_start_time: str = pydantic.Field(..., title="学习开始时间", description="请输入 YYYY-MM-DD 格式的日期")
    study_end_time: str = pydantic.Field(..., title="学习结束时间", description="请输入 YYYY-MM-DD 格式的日期")
    school: str = pydantic.Field(..., title="学校", description="请输入学校全称")
    degree: str = pydantic.Field(..., title="学历/学位", description="请输入学历或学位名称")
    major: str = pydantic.Field(..., title="专业", description="请输入专业名称")

class ResumeJson(BaseModel):
    name: str = pydantic.Field(..., title="姓名", description="请输入姓名")
    age: int = pydantic.Field(..., title="年龄", description="请输入年龄", ge=16)
    birth: str = pydantic.Field(..., title="出生日期", description="请输入出生日期")
    gender: str = pydantic.Field(..., title="性别", description="请输入性别", regex=r"男|女")
    contact: str = pydantic.Field(..., title="联系方式", description="请输入手机号码", regex=r"^\d{11}$")
    residence: str = pydantic.Field(..., title="现居住地", description="请输入现居住地")
    work_experience: List[WorkExperience] = pydantic.Field(..., title="工作经验", description="请输入工作经验列表")
    education_experience: List[EducationExperience] = pydantic.Field(..., title="教育经历", description="请输入教育经历列表")
    class Config:
        schema_extra = {
            "example": {
                "姓名": "黎智豪",
                "年龄": "31岁",
                "出生日期":"1986年1月13日",
                "性别": "男",
                "联系方式": "13580442780",
                "现居住地": "广州-荔湾区",
                "工作经验": [
                    {
                        "工作开始时间": "2015年7月",
                        "工作结束时间": "2016年6月",
                        "公司": "广发银行营销中心",
                        "行业": "无",
                        "职位": "电话销售",
                        "工作内容": "财智金销售"
                    },
                    {
                        "工作开始时间": "2012年4月",
                        "工作结束时间": "2013年3月",
                        "公司": "安利(中国)日用品有限公司",
                        "行业": "美容/保健/外资(欧美)",
                        "职位": "全国业务部客服专员/助理",
                        "工作内容": "负责解答营销人员及顾客关于产品或者业务上的疑难问题"
                    },
                    {
                        "工作开始时间": "2011年3月",
                        "工作结束时间": "2012年3月",
                        "公司": "TVB香港电视有限公司广州分公司",
                        "行业": "生活服务|500-1000人|外资(非欧美)",
                        "职位": "账项跟催部催收专员"
                    }
                ],
                "教育经历": [
                    {
                        "学习开始时间": "2005年9月",
                        "学习结束时间": "2008年7月",
                        "学校": "广东技术师范学院",
                        "学历/学位": "大专",
                        "专业": "贸易经济"
                    }
                ]
            }
        }





class ResumeMessage(BaseModel):
    # question: str = pydantic.Field(..., description="Question text")
    response: ResumeJson = pydantic.Field(..., description="Resume message in json format")
    # history: List[List[str]] = pydantic.Field(..., description="History text")
    # source_documents: List[str] = pydantic.Field(
    #     ..., description="List of source documents and their scores"
    # )





def get_folder_path(local_doc_id: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "content")


def get_vs_path(local_doc_id: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "vector_store")


def get_file_path(local_doc_id: str, doc_name: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "content", doc_name)


async def upload_file(
        file: UploadFile = File(description="A single binary file"),
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):
    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    file_content = await file.read()  # 读取上传文件的内容

    file_path = os.path.join(saved_path, file.filename)
    if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
        file_status = f"文件 {file.filename} 已存在。"
        return BaseResponse(code=200, msg=file_status)

    with open(file_path, "wb") as f:
        f.write(file_content)

    vs_path = get_vs_path(knowledge_base_id)
    vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store([file_path], vs_path)
    if len(loaded_files) > 0:
        file_status = f"文件 {file.filename} 已上传至新的知识库，并已加载知识库，请开始提问。"
        return BaseResponse(code=200, msg=file_status)
    else:
        file_status = "文件上传失败，请重新上传"
        return BaseResponse(code=500, msg=file_status)




async def upload_files(
        files: Annotated[
            List[UploadFile], File(description="Multiple files as UploadFile")
        ],
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):
    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    filelist = []
    for file in files:
        file_content = ''
        file_path = os.path.join(saved_path, file.filename)
        file_content = file.file.read()
        if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
            continue
        with open(file_path, "ab+") as f:
            f.write(file_content)
        filelist.append(file_path)
    if filelist:
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, get_vs_path(knowledge_base_id))
        if len(loaded_files):
            file_status = f"documents {', '.join([os.path.split(i)[-1] for i in loaded_files])} upload success"
            return BaseResponse(code=200, msg=file_status)
    file_status = f"documents {', '.join([os.path.split(i)[-1] for i in loaded_files])} upload fail"
    return BaseResponse(code=500, msg=file_status)


async def list_kbs():
    # Get List of Knowledge Base
    if not os.path.exists(KB_ROOT_PATH):
        all_doc_ids = []
    else:
        all_doc_ids = [
            folder
            for folder in os.listdir(KB_ROOT_PATH)
            if os.path.isdir(os.path.join(KB_ROOT_PATH, folder))
               and os.path.exists(os.path.join(KB_ROOT_PATH, folder, "vector_store", "index.faiss"))
        ]

    return ListDocsResponse(data=all_doc_ids)


async def list_docs(
        knowledge_base_id: Optional[str] = Query(default=None, description="Knowledge Base Name", example="kb1")
):
    local_doc_folder = get_folder_path(knowledge_base_id)
    if not os.path.exists(local_doc_folder):
        return {"code": 1, "msg": f"Knowledge base {knowledge_base_id} not found"}
    all_doc_names = [
        doc
        for doc in os.listdir(local_doc_folder)
        if os.path.isfile(os.path.join(local_doc_folder, doc))
    ]
    return ListDocsResponse(data=all_doc_names)


async def delete_kb(
        knowledge_base_id: str = Query(...,
                                       description="Knowledge Base Name",
                                       example="kb1"),
):
    # TODO: 确认是否支持批量删除知识库
    knowledge_base_id = urllib.parse.unquote(knowledge_base_id)
    if not os.path.exists(get_folder_path(knowledge_base_id)):
        return {"code": 1, "msg": f"Knowledge base {knowledge_base_id} not found"}
    shutil.rmtree(get_folder_path(knowledge_base_id))
    return BaseResponse(code=200, msg=f"Knowledge Base {knowledge_base_id} delete success")


async def delete_doc(
        knowledge_base_id: str = Query(...,
                                       description="Knowledge Base Name",
                                       example="kb1"),
        doc_name: str = Query(
            None, description="doc name", example="doc_name_1.pdf"
        ),
):
    knowledge_base_id = urllib.parse.unquote(knowledge_base_id)
    if not os.path.exists(get_folder_path(knowledge_base_id)):
        return {"code": 1, "msg": f"Knowledge base {knowledge_base_id} not found"}
    doc_path = get_file_path(knowledge_base_id, doc_name)
    if os.path.exists(doc_path):
        os.remove(doc_path)
        remain_docs = await list_docs(knowledge_base_id)
        if len(remain_docs.data) == 0:
            shutil.rmtree(get_folder_path(knowledge_base_id), ignore_errors=True)
            return BaseResponse(code=200, msg=f"document {doc_name} delete success")
        else:
            status = local_doc_qa.delete_file_from_vector_store(doc_path, get_vs_path(knowledge_base_id))
            if "success" in status:
                return BaseResponse(code=200, msg=f"document {doc_name} delete success")
            else:
                return BaseResponse(code=1, msg=f"document {doc_name} delete fail")
    else:
        return BaseResponse(code=1, msg=f"document {doc_name} not found")


async def update_doc(
        knowledge_base_id: str = Query(...,
                                       description="知识库名",
                                       example="kb1"),
        old_doc: str = Query(
            None, description="待删除文件名，已存储在知识库中", example="doc_name_1.pdf"
        ),
        new_doc: UploadFile = File(description="待上传文件"),
):
    knowledge_base_id = urllib.parse.unquote(knowledge_base_id)
    if not os.path.exists(get_folder_path(knowledge_base_id)):
        return {"code": 1, "msg": f"Knowledge base {knowledge_base_id} not found"}
    doc_path = get_file_path(knowledge_base_id, old_doc)
    if not os.path.exists(doc_path):
        return BaseResponse(code=1, msg=f"document {old_doc} not found")
    else:
        os.remove(doc_path)
        delete_status = local_doc_qa.delete_file_from_vector_store(doc_path, get_vs_path(knowledge_base_id))
        if "fail" in delete_status:
            return BaseResponse(code=1, msg=f"document {old_doc} delete failed")
        else:
            saved_path = get_folder_path(knowledge_base_id)
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)

            file_content = await new_doc.read()  # 读取上传文件的内容

            file_path = os.path.join(saved_path, new_doc.filename)
            if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
                file_status = f"document {new_doc.filename} already exists"
                return BaseResponse(code=200, msg=file_status)

            with open(file_path, "wb") as f:
                f.write(file_content)

            vs_path = get_vs_path(knowledge_base_id)
            vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store([file_path], vs_path)
            if len(loaded_files) > 0:
                file_status = f"document {old_doc} delete and document {new_doc.filename} upload success"
                return BaseResponse(code=200, msg=file_status)
            else:
                file_status = f"document {old_doc} success but document {new_doc.filename} upload fail"
                return BaseResponse(code=500, msg=file_status)



async def local_doc_chat(
        knowledge_base_id: str = Body(..., description="Knowledge Base Name", example="kb1"),
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    vs_path = get_vs_path(knowledge_base_id)
    if not os.path.exists(vs_path):
        # return BaseResponse(code=1, msg=f"Knowledge base {knowledge_base_id} not found")
        return ChatMessage(
            question=question,
            response=f"Knowledge base {knowledge_base_id} not found",
            history=history,
            source_documents=[],
        )
    else:
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=question, vs_path=vs_path, chat_history=history, streaming=True
        ):
            pass
        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"])
        ]

        return ChatMessage(
            question=question,
            response=resp["result"],
            history=history,
            source_documents=source_documents,
        )


async def bing_search_chat(
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: Optional[List[List[str]]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    for resp, history in local_doc_qa.get_search_result_based_answer(
            query=question, chat_history=history, streaming=True
    ):
        pass
    source_documents = [
        f"""出处 [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n"""
        for inum, doc in enumerate(resp["source_documents"])
    ]

    return ChatMessage(
        question=question,
        response=resp["result"],
        history=history,
        source_documents=source_documents,
    )


async def chat(
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    for answer_result in local_doc_qa.llm.generatorAnswer(prompt=question, history=history,
                                                          streaming=True):
        resp = answer_result.llm_output["answer"]
        history = answer_result.history
        pass

    return ChatMessage(
        question=question,
        response=resp,
        history=history,
        source_documents=[],
    )









async def stream_chat(websocket: WebSocket, knowledge_base_id: str):
    await websocket.accept()
    turn = 1
    while True:
        input_json = await websocket.receive_json()
        question, history, knowledge_base_id = input_json["question"], input_json["history"], input_json[
            "knowledge_base_id"]
        vs_path = get_vs_path(knowledge_base_id)

        if not os.path.exists(vs_path):
            await websocket.send_json({"error": f"Knowledge base {knowledge_base_id} not found"})
            await websocket.close()
            return

        await websocket.send_json({"question": question, "turn": turn, "flag": "start"})

        last_print_len = 0
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=question, vs_path=vs_path, chat_history=history, streaming=True
        ):
            await asyncio.sleep(0)
            await websocket.send_text(resp["result"][last_print_len:])
            last_print_len = len(resp["result"])

        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"])
        ]

        await websocket.send_text(
            json.dumps(
                {
                    "question": question,
                    "turn": turn,
                    "flag": "end",
                    "sources_documents": source_documents,
                },
                ensure_ascii=False,
            )
        )
        turn += 1


async def document():
    return RedirectResponse(url="/docs")






json_schema = {
        "姓名": "",
        "年龄": "",
        "性别": "",
        "联系方式":"",
        "现居住地": "",
        "工作经验": [
            {   "工作开始时间": "",
                "工作结束时间": "",
                "公司": "",
                "行业": "",
                "职位": "",
                "工作内容": ""
            },
        ...
        ],
        "教育经历": [
            {   "学习开始时间": "",
                "学习结束时间": "",
                "学校": "",
                "学历/学位": "",
                "专业": ""
            },...
        ]
}


# Define the schema for each field in the "工作经验" list
work_experience_schema = ResponseSchema(
    name="工作经验",
    description="This represents a single work experience item.",
    children=[
        ResponseSchema(name="工作开始时间", description="工作开始时间"),
        ResponseSchema(name="工作结束时间", description="工作结束时间"),
        ResponseSchema(name="公司", description="公司"),
        ResponseSchema(name="行业", description="行业"),
        ResponseSchema(name="职位", description="职位"),
        ResponseSchema(name="工作内容", description="工作内容"),
    ]
)

# Define the schema for each field in the "教育经历" list
education_experience_schema = ResponseSchema(
    name="教育经历",
    description="This represents a single education experience item.",
    children=[

        ResponseSchema(name="学校", description="学校"),
        ResponseSchema(name="学历/学位", description="学历/学位"),
        ResponseSchema(name="专业", description="专业"),
        ResponseSchema(name="学习开始时间", description="学习开始时间"),
        ResponseSchema(name="学习结束时间", description="学习结束时间"),
    ]
)

# Now, define the overall schema
name_schema = ResponseSchema(name="姓名", description="Extract the name.")
gender_schema = ResponseSchema(name="性别", description="Extract the gender.")
age_schema = ResponseSchema(name="年龄", description="Extract the age.")
education_level_schema = ResponseSchema(name="学历", description="Extract the education level.")
graduated_school_schema = ResponseSchema(name="毕业院校", description="Extract the graduated school.")
phone_number_schema = ResponseSchema(name="联系电话", description="Extract the phone number.")
work_experience_schema = ResponseSchema(name="工作经历", description="Extract the work experiences.")
education_experience_schema = ResponseSchema(name="教育经历", description="Extract the education experiences.")


response_schemas=[
    name_schema, gender_schema, age_schema, education_level_schema, graduated_school_schema, phone_number_schema, work_experience_schema, education_experience_schema
    ]

# outputparser = StructuredOutputParser.from_response_schemas(response_schemas)
# logging.info("outputparser:/n")
# logging.info(outputparser)
@app.post("/ask_resume")
async def ask_resume(
        file: UploadFile = File(description="A single binary file")
):
    # Generate a unique knowledge_base_id based on the current timestamp
    knowledge_base_id = str(uuid.uuid4())

    # Define the fixed question
    # question='''
    # 按照{json_schema}生成json格式简历信息，其中联系方式是11位数字的手机号码

    # '''.format(
    # json_schema=json_schema
    # )   

  

    # resumeparser = PydanticOutputParser(pydantic_object=ResumeJson)
    # output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    # prompt = PromptTemplate(
    # template="Extract information from the local resume.\n{format_instructions}\n{query}\n",
    # input_variables=["query"],
    # partial_variables={"format_instructions": resumeparser.get_format_instructions()},
    # )

    # query = "Please extract the name, age, birth, gender, contact, residence, work experience and education experience from the resume."
    # prompt_value = prompt.format_prompt(query=query)

    # question=query
    # Upload the file,ChatMessage 的 response 改成response=f"文件上传失败，请重新上传"之类的,

    # outputparser = StructuredOutputParser.from_response_schemas(response_schemas)
    # logging.info("outputparser:/n")
    # logging.info(outputparser)

    # format_instructions = outputparser.get_format_instructions()
    question='extract information from resume'

    # response_schemas=[
    # name_schema, gender_schema, age_schema, education_level_schema, graduated_school_schema, phone_number_schema, work_experience_schema, education_experience_schema
    # ]

    # template="{question}\n{format_instructions}"
    # prompt = ChatPromptTemplate.from_template(template=template)

    # messages = prompt.format_messages(
    #                             format_instructions=format_instructions)


    upload_response = await upload_file(file=file, knowledge_base_id=knowledge_base_id)


    if upload_response.code != 200:
        return ChatMessage(question=question, response=upload_response.msg, history=[], source_documents=[])
        

    vs_path = get_vs_path(knowledge_base_id)
    if not os.path.exists(vs_path):
        return BaseResponse(code=1, msg=f"Knowledge base {knowledge_base_id} not found")
        

  
    else:
        for resp, history in local_doc_qa.get_json_output_local_file_based_answer(
            response_schemas=response_schemas,query=question, vs_path=vs_path, chat_history=[], streaming=True
        ):
            pass
        # return ResumeMessage(
        #     response=resp['result']
        # )
        # logging.info("response results:/n")
        
       

        # logging.info(resp['result'])
        # logging.info(type(resp['result']))

        # response_= resp['result']

        # text=response_.content
        # logging.info('text:/n')
        # logging.info(text)


        # messages = prompt.format_messages(resume=text, 
        #                         format_instructions=format_instructions)

        
        # logging.info('resumeparser:/n')
        # logging.info(resumeparser.parse(resp['result']))


        return resp['result']

    # Ask the question
    # chat_response = await local_doc_chat(knowledge_base_id=knowledge_base_id, question=question, history=[])
    # return chat_response




# app = FastAPI()

def api_start(host, port, **kwargs):
    # global app
    # global local_doc_qa

    # llm_model_ins = shared.loaderLLM()
    # llm_model_ins.set_history_len(LLM_HISTORY_LEN)

    # # app = FastAPI()
    # # Add CORS middleware to allow all origins
    # # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # # set OPEN_DOMAIN=True in config.py to allow cross-domain
    # if OPEN_CROSS_DOMAIN:
    #     app.add_middleware(
    #         CORSMiddleware,
    #         allow_origins=["*"],
    #         allow_credentials=True,
    #         allow_methods=["*"],
    #         allow_headers=["*"],
    #     )
    # app.websocket("/local_doc_qa/stream-chat/{knowledge_base_id}")(stream_chat)

    # app.get("/", response_model=BaseResponse)(document)

    # app.post("/chat", response_model=ChatMessage)(chat)

    # app.post("/local_doc_qa/upload_file", response_model=BaseResponse)(upload_file)
    # app.post("/local_doc_qa/upload_files", response_model=BaseResponse)(upload_files)
    # app.post("/local_doc_qa/local_doc_chat", response_model=ChatMessage)(local_doc_chat)
    # app.post("/local_doc_qa/bing_search_chat", response_model=ChatMessage)(bing_search_chat)
    # app.get("/local_doc_qa/list_knowledge_base", response_model=ListDocsResponse)(list_kbs)
    # app.get("/local_doc_qa/list_files", response_model=ListDocsResponse)(list_docs)
    # app.delete("/local_doc_qa/delete_knowledge_base", response_model=BaseResponse)(delete_kb)
    # app.delete("/local_doc_qa/delete_file", response_model=BaseResponse)(delete_doc)
    # app.post("/local_doc_qa/update_file", response_model=BaseResponse)(update_doc)
    # # app.post("/ask_resume", response_model=ResumeMessage)(ask_resume)
    # app.post("/ask_resume")(ask_resume)

    # local_doc_qa = LocalDocQA()
    # local_doc_qa.init_cfg(
    #     llm_model=llm_model_ins,
    #     embedding_model=EMBEDDING_MODEL,
    #     embedding_device=EMBEDDING_DEVICE,
    #     top_k=VECTOR_SEARCH_TOP_K,
    # )
    # development stage
    if kwargs.get("reload"):
        uvicorn.run("api:app", host=host, port=port, reload=kwargs.get("reload"))
    # production stage
    else:
        uvicorn.run(app, host=host, port=port)



if __name__ == "__main__":
    
    # parser.add_argument("--host", type=str, default="0.0.0.0")
    # parser.add_argument("--port", type=int, default=7860)
    # parser.add_argument("--reload",type=bool,default=True)
    # parser.add_argument("--timeout-keep-alive",type=int,default=5000)
    # 初始化消息
    # args = None
    # args = parser.parse_args()
    # args_dict = vars(args)
    # shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    # api_start(args.host, args.port, reload=args.reload)
    

    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)
    # uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=True,timeout_keep_alive=5000 )    
