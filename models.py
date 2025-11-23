from sqlalchemy import Column, Integer, String, Boolean, Numeric, Date
from database import Base

class VehicleSold(Base):
    __tablename__ = "vehicles_sold"

    id = Column(Integer, primary_key=True, index=True)
    state = Column(String, nullable=True)
    company = Column(String, nullable=True)
    vin = Column(String, nullable=False, unique=True, index=True)
    year = Column(Integer, nullable=False)
    make = Column(String, nullable=False)
    model = Column(String, nullable=False)
    mileage = Column(Integer, nullable=True)
    has_keys = Column(String, nullable=True)
    runs = Column(String, nullable=True)
    drives = Column(String, nullable=True)
    sale_price = Column(Numeric, nullable=False)
    sale_date = Column(Date, nullable=True)
    title_status = Column(String, nullable=True)
    source_file = Column(String, nullable=True)

