from sqlalchemy.orm import Session

from database.models import Job


def get_jobs_processed(db: Session) -> int:

    return (
        db.query(Job)
        .filter(Job.status == "completed")
        .count()
    )


def get_jobs_running(db: Session) -> int:

    return (
        db.query(Job)
        .filter(Job.status == "running")
        .count()
    )


def get_jobs_queued(db: Session) -> int:

    return (
        db.query(Job)
        .filter(Job.status == "queued")
        .count()
    )