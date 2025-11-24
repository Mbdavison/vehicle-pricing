from sqlalchemy import Column, Integer, String, Float, Date
from database import Base


class Vehicle(Base):
    __tablename__ = "vehicles_sold"

    id = Column(Integer, primary_key=True, index=True)

    vin = Column(String, index=True, nullable=False)

    year = Column(Integer, index=True, nullable=True)
    make = Column(String, index=True, nullable=True)
    model = Column(String, index=True, nullable=True)

    mileage = Column(Integer, nullable=True)

    # Store whatever text is in the spreadsheet for these:
    has_keys = Column(String, nullable=True)
    runs = Column(String, nullable=True)
    drives = Column(String, nullable=True)

    sale_price = Column(Float, nullable=True)
    sale_date = Column(Date, nullable=True)

    title_status = Column(String, nullable=True)

    source_file = Column(String, nullable=True)

    # New fields for segmentation:
    state = Column(String, index=True, nullable=True)
    company = Column(String, index=True, nullable=True)

