from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler()

@sched.scheduled_job('cron', day_of_week='mon-sun', hour='12', minute='00')
def timed_job():
    print("---------- JOB START ----------")
    import daily_job
    print("---------- JOB OVER ----------")


sched.start()
