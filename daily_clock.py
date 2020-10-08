from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler()

@sched.scheduled_job('interval', hour='15', minute='30')
def timed_job():
    import daily_job

sched.start()
