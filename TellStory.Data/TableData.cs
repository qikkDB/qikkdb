using System;
using System.Data;

namespace TellStory.Data
{
    public class TableData : StructuredData
    {
        private DataTable dt = new DataTable();

        public TableData() : base() { }

        public TableData(string name) : base(name) { }

        public void AddColumn(string columnName, Type type)
        {
            dt.Columns.Add(columnName, type);
        }

        public void AddRow(params object[] values)
        {
            dt.Rows.Add(values);
        }

        public object[] GetRow(int index)
        {
            return dt.Rows[index].ItemArray;
        }

        public long GetCount()
        {
            return dt.Rows.Count;
        }

        public DataColumnCollection GetColumns()
        {
            return dt.Columns;
        }

        public DataTable GetData()
        {
            return dt;
        }

        public void Clear()
        {
            dt.Clear();
        }

    }
}
