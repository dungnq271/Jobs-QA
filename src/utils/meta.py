from typing import Any

from .func import modify_days_to_3digits

CHUNKING_REGEX = r"([^,.;。？！]+(?:http)*.*)[,.;。？！]?"


# File Metadata
metadata: dict[str, str | dict[str, Any]] = {
    "table_name": "jobPosted",
    "file_description": "different jobs information at different companies",
    "renamed_column": {
        "Posted": "Number_of_days_posted_ago",
        "Salary": "Pay",
    },
    "column_map_function": {
        "Number_of_days_posted_ago": modify_days_to_3digits
    },
    "column_description": {
        "Logo": {"type": "str", "description": None},
        "Role": {"type": "str", "description": None},
        "Company": {"type": "str", "description": None},
        "Location": {"type": "str", "description": None},
        "Source": {"type": "str", "description": None},
        "Posted": {
            "type": "str",
            "description": "The number of days ago the job was posted",
        },
        "Full / Part Time": {
            "type": "str",
            # "description": "Working time for the job, including "
            # "Fulltime, Partime, Internship, combination of these, "
            # "and these in Vietnamese",
            "description": "Type of work arrangement associated with each "
            "job, including Full-time, Part-time, Internship, "
            "or any combination thereof. It may also include the "
            "corresponding terms in Vietnamese, such as 'Toàn thời gian' "
            "for Full-time",
        },
        "Salary": {
            "type": "str",
            "description": "The pay range of the job",
        },
        "Description": {
            "type": "str",
            "description": "The detailed description of the job",
        },
        "Link": {
            "type": "str",
            "description": "Link to the posted job",
        },
    },
}
