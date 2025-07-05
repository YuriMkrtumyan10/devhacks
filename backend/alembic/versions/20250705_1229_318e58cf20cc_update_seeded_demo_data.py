
"""update seeded demo data

Revision ID: 318e58cf20cc
Revises: b52a26b52ff2
Create Date: 2025-07-05 12:29:22.897087+00:00

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers, used by Alembic.
revision = '318e58cf20cc'
down_revision = 'b52a26b52ff2'
branch_labels = None
depends_on = None

def upgrade() -> None:
    """
    UPDATE (not INSERT) the seed rows that were introduced in 0cc84414ad86.
    Use parameter binding to stay safe and DB-portable.
    """
    conn = op.get_bind()

    # Acme Inc. — add missing columns
    conn.execute(
        sa.text(
            """
            UPDATE companies
            SET country       = :country,
                city          = :city,
                contact_email = :email,
                founded_year  = :year,
                size          = :size,
                updated_at    = :ts             -- if you track it
            WHERE name = 'Acme Inc.'
            """
        ),
        {
            "country": "United States",
            "city":    "New York",
            "email":   "contact@acme.com",
            "year":    1985,
            "size":    "501-1000",
            "ts":      datetime.utcnow(),
        },
    )

    # Globex LLC — add missing columns
    conn.execute(
        sa.text(
            """
            UPDATE companies
            SET country       = :country,
                city          = :city,
                contact_email = :email,
                founded_year  = :year,
                size          = :size,
                updated_at    = :ts
            WHERE name = 'Globex LLC'
            """
        ),
        {
            "country": "Canada",
            "city":    "Toronto",
            "email":   "info@globex.com",
            "year":    1995,
            "size":    "201-500",
            "ts":      datetime.utcnow(),
        },
    )

    # Example: rotate the super-admin password hash
    conn.execute(
        sa.text(
            """
            UPDATE users
            SET password_hash = :new_hash,
                updated_at    = :ts
            WHERE email = 'superadmin@example.com'
            """
        ),
        {
            "new_hash": "$2b$12$newHashGoesHere.........",
            "ts":       datetime.utcnow(),
        },
    )


def downgrade() -> None:
    """
    Revert ONLY the columns you touched above, so rollbacks are clean.
    """
    conn = op.get_bind()

    # Acme Inc. revert
    conn.execute(
        sa.text(
            """
            UPDATE companies
            SET country       = NULL,
                city          = NULL,
                contact_email = NULL,
                founded_year  = NULL,
                size          = NULL,
                updated_at    = NULL
            WHERE name = 'Acme Inc.'
            """
        )
    )

    # Globex LLC revert
    conn.execute(
        sa.text(
            """
            UPDATE companies
            SET country       = NULL,
                city          = NULL,
                contact_email = NULL,
                founded_year  = NULL,
                size          = NULL,
                updated_at    = NULL
            WHERE name = 'Globex LLC'
            """
        )
    )

    # Super-admin password revert (put back the old hash or NULL)
    conn.execute(
        sa.text(
            """
            UPDATE users
            SET password_hash = '$2b$12$QEbMv.jDw0X8M2c/8C1I5ex9znAy0lP3JzqFZqT1k4bH3gMtOgd8K',
                updated_at    = NULL
            WHERE email = 'superadmin@example.com'
            """
        )
    )
