import mysql
from mysql.connector import (connection)
from source_code.sec import secureLog as SL 


class Sqldd():
    """Class return connection & cursor for bdd,you need to close them 
    Instantiate :   tip = Sqldd()
                    cnx, cursor = tip.get_bdd_co()
    """

    def __init__(self) :
        
        self.log_user = SL.sqlLogUser
        self.log_pass = SL.sqlLogPass
        self.log_host = SL.sqlLogHost
        self.log_bdd = SL.sqlLogDatabase
        self.log_port = SL.sqlLogPort
    
    def get_bdd_co(self):
        cnx = mysql.connector.connect(user=f'{self.log_user}', password=f'{self.log_pass}',
                              host=f'{self.log_host}',
                              database=f'{self.log_bdd}', port=f'{self.log_port}')

        cnx = connection.MySQLConnection(user=f'{self.log_user}', password=f'{self.log_pass}',
                              host=f'{self.log_host}',
                              database=f'{self.log_bdd}', port=f'{self.log_port}')

        cursor = cnx.cursor(named_tuple=True)

        return cnx, cursor