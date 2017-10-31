from sqlalchemy import Column, String, create_engine, Integer, DECIMAL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class User(Base):
    __tablename__ = 'user'

    id = Column(Integer(), primary_key=True)
    open = Column(DECIMAL())
    high = Column(DECIMAL())
    low = Column(DECIMAL())
    close = Column(DECIMAL())
    ratio = Column(DECIMAL())
    a = Column(DECIMAL())


engine = create_engine('mysql+mysqlconnector://root:password@localhost:3306/test')
DBSession = sessionmaker(bind=engine)

session = DBSession()
new_user = User(id='5', name='Bob')

user = session.query(User).filter(User.id == '5').one()

session.add(new_user)
session.commit()
session.close()


