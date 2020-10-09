from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler()

@sched.scheduled_job('cron',day_of_week='mon-sun', hour='14', minute='25', timezone="France/Paris")
def timed_job():
    import daily_job
    print("---------- JOB OVER ----------")


@sched.scheduled_job('interval', minutes=1)
def timed_job_2():
    print(' *** - *** This job is run every one minutes. *** - ***')

sched.start()
