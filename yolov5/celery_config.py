# celery_config.py
import celery

# 执行如下命令: celery -A celery_object worker -l info

backend = "redis://127.0.0.1:6379/4"  # 设置redis的4号数据库来存放结果
broker = "redis://127.0.0.1:6379/5"  # 设置redis的5号数据库存放消息中间件
celery_app = celery.Celery(
    "celery_demo",
    backend=backend,
    broker=broker,
    include=[
        "celery_task",
    ],
)

celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]

celery_app.conf.timezone = "Asia/Shanghai"  # 时区
celery_app.conf.enable_utc = False  # 是否使用UTC
