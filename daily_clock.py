from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler()

@sched.scheduled_job('cron',day_of_week='mon-sun', hour='16', minute='05')
def timed_job():
    import daily_job

sched.start()
