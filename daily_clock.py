from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler()

@sched.scheduled_job('cron', day_of_week='mon-sun', hour='12', minute='36')
def timed_job():
    print("---------- JOB START ----------")
    import daily_job
    print("---------- JOB OVER ----------")


@sched.scheduled_job('interval', minutes=5)
def timed_job_2():
    print(' *** - *** This job is run every one minutes. *** - ***')

sched.start()
