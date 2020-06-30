using System;
using System.Collections.Generic;
using System.Text;

namespace QikkDB.NetworkClient
{
    /// <summary>
    /// Represents an error, that occured during query execution or during data import
    /// </summary>
    public class QueryException : Exception
    {
        public QueryException(string message) :
            base(message)
        {            
        }
    }
}
