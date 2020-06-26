#include "gtest/gtest.h"
#include "../qikkDB/Database.h"
#include "../qikkDB/CSVDataImporter.h"
#include "../qikkDB/DataType.h"
#include "../qikkDB/ColumnBase.h"


TEST(CSVDataImportTests, CreateTable)
{
	auto database = std::make_shared<Database>("testDatabase1", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("csv_tests/valid_header.csv", true, ',');
	importer.ImportTables(database);

	ASSERT_EQ(true, database->GetTables().find("valid_header") != database->GetTables().end());
	Database::RemoveFromInMemoryDatabaseList("testDatabase1");
}

TEST(CSVDataImportTests, ImportHeader)
{
	auto database = std::make_shared<Database>("testDatabase2", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("csv_tests/valid_header.csv", true, ',');
	importer.ImportTables(database);
	
	ASSERT_EQ(true, database->GetTables().find("valid_header")->second.GetColumns().find("longitude") != database->GetTables().find("valid_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_header")->second.GetColumns().find("latitude") != database->GetTables().find("valid_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_header")->second.GetColumns().find("targetId") != database->GetTables().find("valid_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_header")->second.GetColumns().find("genderId") != database->GetTables().find("valid_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_header")->second.GetColumns().find("ageId") != database->GetTables().find("valid_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_header")->second.GetColumns().find("wealthIndexId") != database->GetTables().find("valid_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_header")->second.GetColumns().find("hwOsId") != database->GetTables().find("valid_header")->second.GetColumns().end());
	Database::RemoveFromInMemoryDatabaseList("testDatabase2");
}

TEST(CSVDataImportTests, ImportWithoutHeader)
{
	auto database = std::make_shared<Database>("testDatabase3", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("csv_tests/valid_no_header.csv", false, ',');
	importer.ImportTables(database);

	ASSERT_EQ(true, database->GetTables().find("valid_no_header")->second.GetColumns().find("C0") != database->GetTables().find("valid_no_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_no_header")->second.GetColumns().find("C1") != database->GetTables().find("valid_no_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_no_header")->second.GetColumns().find("C2") != database->GetTables().find("valid_no_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_no_header")->second.GetColumns().find("C3") != database->GetTables().find("valid_no_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_no_header")->second.GetColumns().find("C4") != database->GetTables().find("valid_no_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_no_header")->second.GetColumns().find("C5") != database->GetTables().find("valid_no_header")->second.GetColumns().end());
	ASSERT_EQ(true, database->GetTables().find("valid_no_header")->second.GetColumns().find("C6") != database->GetTables().find("valid_no_header")->second.GetColumns().end());
	Database::RemoveFromInMemoryDatabaseList("testDatabase3");
}

TEST(CSVDataImportTests, GuessTypes)
{
	auto database = std::make_shared<Database>("testDatabase4", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("csv_tests/valid_header.csv", true, ',');
	importer.ImportTables(database);

	ASSERT_EQ(COLUMN_FLOAT, database->GetTables().find("valid_header")->second.GetColumns().find("longitude")->second->GetColumnType());
	ASSERT_EQ(COLUMN_FLOAT, database->GetTables().find("valid_header")->second.GetColumns().find("latitude")->second->GetColumnType());
	ASSERT_EQ(COLUMN_INT, database->GetTables().find("valid_header")->second.GetColumns().find("targetId")->second->GetColumnType());
	ASSERT_EQ(COLUMN_INT, database->GetTables().find("valid_header")->second.GetColumns().find("genderId")->second->GetColumnType());
	ASSERT_EQ(COLUMN_INT, database->GetTables().find("valid_header")->second.GetColumns().find("ageId")->second->GetColumnType());
	ASSERT_EQ(COLUMN_INT, database->GetTables().find("valid_header")->second.GetColumns().find("wealthIndexId")->second->GetColumnType());
	ASSERT_EQ(COLUMN_INT, database->GetTables().find("valid_header")->second.GetColumns().find("hwOsId")->second->GetColumnType());
	Database::RemoveFromInMemoryDatabaseList("testDatabase4");
}

TEST(CSVDataImportTests, GuessTypesMessedTypes)
{
	auto database = std::make_shared<Database>("testDatabase5", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("csv_tests/valid_header_messed_types.csv", true, ',');
	importer.ImportTables(database);

	ASSERT_EQ(COLUMN_STRING, database->GetTables().find("valid_header_messed_types")->second.GetColumns().find("longitude")->second->GetColumnType());
	ASSERT_EQ(COLUMN_FLOAT, database->GetTables().find("valid_header_messed_types")->second.GetColumns().find("latitude")->second->GetColumnType());
	ASSERT_EQ(COLUMN_LONG, database->GetTables().find("valid_header_messed_types")->second.GetColumns().find("targetId")->second->GetColumnType());
	ASSERT_EQ(COLUMN_FLOAT, database->GetTables().find("valid_header_messed_types")->second.GetColumns().find("genderId")->second->GetColumnType());	
	ASSERT_EQ(COLUMN_STRING, database->GetTables().find("valid_header_messed_types")->second.GetColumns().find("ageId")->second->GetColumnType());
	Database::RemoveFromInMemoryDatabaseList("testDatabase5");
}

TEST(CSVDataImportTests, ImportSingleThread)
{
	auto database = std::make_shared<Database>("testDatabase6", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("csv_tests/valid_header.csv", true, ',');
	importer.SetNumberOfThreads(1);
	importer.ImportTables(database);



	ASSERT_EQ(101, dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("valid_header")->second.GetColumns().at("targetId").get())->GetBlocksList().front()->GetSize());
	ASSERT_EQ(11, dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("valid_header")->second.GetColumns().at("targetId").get())->GetBlocksList().front()->GetData()[10]);
	ASSERT_EQ(21.2282657634477f, dynamic_cast<ColumnBase<float>*>(database->GetTables().find("valid_header")->second.GetColumns().at("longitude").get())->GetBlocksList().front()->GetData()[11]);
	ASSERT_EQ(-1, dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("valid_header")->second.GetColumns().at("genderId").get())->GetBlocksList().front()->GetData()[12]);
	ASSERT_EQ(3, dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("valid_header")->second.GetColumns().at("hwOsId").get())->GetBlocksList().front()->GetData()[100]);
	Database::RemoveFromInMemoryDatabaseList("testDatabase6");
}

TEST(CSVDataImportTests, ImportMultiThread)
{
	auto database = std::make_shared<Database>("testDatabase7", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("csv_tests/valid_header.csv", true, ',');
	importer.ImportTables(database);



	ASSERT_EQ(101, dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("valid_header")->second.GetColumns().at("targetId").get())->GetBlocksList().front()->GetSize());
	Database::RemoveFromInMemoryDatabaseList("testDatabase7");
}

TEST(CSVDataImportTests, ImportSkipRow)
{
	auto database = std::make_shared<Database>("testDatabase8", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("csv_tests/invalid_row_header.csv", true, ',');
	importer.ImportTables(database);

	ASSERT_EQ(100, dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("invalid_row_header")->second.GetColumns().at("targetId").get())->GetBlocksList().front()->GetSize());	
	Database::RemoveFromInMemoryDatabaseList("testDatabase8");
}

TEST(CSVDataImportTests, WktTypes)
{
	auto database = std::make_shared<Database>("testDatabase9", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("csv_tests/wkt_header.csv", true, ';');
	importer.ImportTables(database);

	ASSERT_EQ(COLUMN_POINT, database->GetTables().find("wkt_header")->second.GetColumns().find("p1")->second->GetColumnType());
	ASSERT_EQ(COLUMN_POLYGON, database->GetTables().find("wkt_header")->second.GetColumns().find("p2")->second->GetColumnType());
	ASSERT_EQ(COLUMN_INT, database->GetTables().find("wkt_header")->second.GetColumns().find("p3")->second->GetColumnType());
	ASSERT_EQ(COLUMN_STRING, database->GetTables().find("wkt_header")->second.GetColumns().find("p4")->second->GetColumnType());
	Database::RemoveFromInMemoryDatabaseList("testDatabase9");
}

TEST(CSVDataImportTests, WktTypesMessedTypes)
{
	auto database = std::make_shared<Database>("testDatabase10", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("csv_tests/wkt_header_messed_types.csv", true, ';');
	importer.ImportTables(database);

	ASSERT_EQ(COLUMN_STRING, database->GetTables().find("wkt_header_messed_types")->second.GetColumns().find("p1")->second->GetColumnType());
	ASSERT_EQ(COLUMN_STRING, database->GetTables().find("wkt_header_messed_types")->second.GetColumns().find("p2")->second->GetColumnType());
	ASSERT_EQ(COLUMN_STRING, database->GetTables().find("wkt_header_messed_types")->second.GetColumns().find("p3")->second->GetColumnType());
	ASSERT_EQ(COLUMN_STRING, database->GetTables().find("wkt_header_messed_types")->second.GetColumns().find("p4")->second->GetColumnType());
	Database::RemoveFromInMemoryDatabaseList("testDatabase10");
}

TEST(CSVDataImportTests, WktImport)
{
	auto database = std::make_shared<Database>("testDatabase11", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("csv_tests/wkt_header.csv", true, ';');
	importer.ImportTables(database);

	ASSERT_EQ(101, dynamic_cast<ColumnBase<QikkDB::Types::Point>*>(database->GetTables().find("wkt_header")->second.GetColumns().at("p1").get())->GetBlocksList().front()->GetSize());
	Database::RemoveFromInMemoryDatabaseList("testDatabase11");
}

TEST(CSVDataImportTests, WktImportInvalidRow)
{
	auto database = std::make_shared<Database>("testDatabase12", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter("csv_tests/wkt_header_invalid_row.csv", true, ';');
	importer.ImportTables(database);

	ASSERT_EQ(100, dynamic_cast<ColumnBase<QikkDB::Types::Point>*>(database->GetTables().find("wkt_header_invalid_row")->second.GetColumns().at("p1").get())->GetBlocksList().front()->GetSize());
	Database::RemoveFromInMemoryDatabaseList("testDatabase12");
}

TEST(CSVDataImportTests, CreateTableFromString)
{
	const auto inputString = "longitude,latitude,targetId,genderId,ageId,wealthIndexId,hwOsId\n\
17.1911813399913, 48.2262068066084, 1, -1, 1, 11, 2\n\
17.1120308046253, 48.1916689725291, 2, -1, 0, 11, 2\n\
17.137051565592, 48.1963514926608, 3, 1, 1, 13, 3\n\
19.1030194809568, 48.7087577518036, 4, -1, 0, 11, 2\n\
19.0297830535662, 48.7446182349808, 5, 1, 1, 12, 2\n\
17.1642015401874, 48.1739705574233, 6, 2, 6, 14, 3\n\
17.1575513430304, 48.0435042254544, 7, -1, 0, 11, 2\n\
18.1108274310214, 48.3230460790016, 8, 2, 5, 14, 3\n\
21.260598051649, 48.9893752781195, 9, -1, 0, -1, 1\n\
17.154569259203, 48.2056729014943, 10, 1, 0, -1, -1\n\
21.2801768571342, 48.7217686380594, 11, -1, 1, 11, 2\n\
21.2282657634477, 49.0175952916438, 12, 1, 1, 12, 2\n\
18.0920688252035, 48.3226431710195, 13, -1, 0, -1, -1\n\
17.1336134639847, 48.157690914556, 14, -1, 0, -1, 1\n\
21.2157960495558, 48.9834671618524, 15, -1, 0, 11, 2\n\
17.1457386025827, 48.136395108473, 16, -1, 0, 11, 2\n\
21.2359807907819, 49.0314101229962, 17, 1, 1, 12, 2\n\
17.032993940167, 48.2393311844409, 18, 1, 1, 12, 2\n\
18.70530088522, 49.2474893771916, 19, -1, 0, 11, 2\n\
19.1296149605427, 48.7354566926547, 20, 1, 5, 13, 3\n\
21.1980822796886, 48.7790385330901, 21, -1, 0, 11, 2\n\
21.2271508085085, 48.7368486690076, 22, 1, 1, 12, 2\n\
17.6022402830144, 48.378249224587, 23, 2, 6, 14, 3\n\
18.7917811019651, 49.243018912244, 24, 1, 1, 12, 2\n\
21.1734535513608, 48.9779014797044, 25, -1, 1, 11, 2\n\
21.2807242670816, 49.0270912557478, 26, 1, 1, 11, 2\n\
19.101353895609, 48.7546941811633, 27, -1, 0, -1, 2\n\
18.0915388196816, 48.2822852650398, 28, -1, 0, 11, 2\n\
17.2206981686957, 48.195116651325, 29, 1, 1, 12, 2\n\
19.040371984722, 48.7273600658011, 30, -1, 0, 11, 2\n\
17.0322670403572, 48.2406644394767, 31, 2, 6, 14, 3\n\
19.1608426128037, 48.7429181440887, 32, 1, 5, 14, 3\n\
18.1186785568142, 48.3085294040525, 33, 2, 6, 14, 7\n\
18.7229753848342, 49.2486112774919, 34, -1, 1, 11, 2\n\
21.2650211369085, 48.9701319418828, 35, 2, 6, 14, 4\n\
21.2636671412281, 48.9721128681997, 36, 1, 1, 11, 2\n\
18.120813123356, 48.3102668536849, 37, 1, 1, 12, 2\n\
17.1510714605932, 48.1675317621538, 38, -1, 0, 11, 2\n\
21.1723143887861, 48.7689671519407, 39, -1, 0, -1, 2\n\
17.1103912723681, 48.0637726295506, 40, -1, 0, 11, 2\n\
17.1670579704554, 48.0163704172166, 41, -1, 0, 11, 2\n\
17.2699352230583, 48.1076503885326, 42, 2, 5, 14, 3\n\
21.2658830052322, 48.6950746034107, 43, -1, 1, 11, 2\n\
19.1737545783225, 48.7376538723141, 44, 2, 5, 14, 3\n\
17.1483632705747, 48.1537431997214, 45, 1, 1, 12, 2\n\
21.1831327986519, 48.7246185575466, 46, -1, 0, 11, 2\n\
17.0767776101525, 48.1578410845609, 47, -1, 0, -1, 1\n\
18.7390742239502, 49.2656559441761, 48, 1, 1, 12, 2\n\
18.0599425459319, 48.3393543010614, 49, -1, 1, 11, 2\n\
21.2988151632228, 48.727472390728, 50, 1, 1, 12, 2\n\
17.104826746637, 48.1690277255266, 51, 2, 5, 14, 3\n\
17.1606796798867, 48.0431871609626, 52, -1, 0, -1, 2\n\
21.2153329789171, 48.739104195403, 53, -1, 0, -1, -1\n\
18.096698803621, 48.3109541860706, 54, -1, 0, -1, -1\n\
21.2537470990847, 48.9579237058612, 55, 1, 1, 13, 3\n\
19.2288265238046, 48.7434867923389, 56, 1, 5, 14, 3\n\
19.1313071451152, 48.7037611304646, 57, -1, 0, 11, 2\n\
19.116268974379, 48.697737866397, 58, -1, 0, -1, -1\n\
17.2200301181474, 48.1407006709587, 59, 1, 5, 14, 3\n\
17.0595892006296, 48.2381512568764, 60, -1, 0, -1, 1\n\
21.1582436189554, 48.6329263151003, 61, 1, 5, 14, 3\n\
17.1178784431167, 48.0932591531887, 62, 2, 6, 14, 6\n\
17.1606132169184, 48.1111278920224, 63, -1, 0, 11, 2\n\
18.069711576972, 48.3056254548109, 64, 2, 5, 14, 3\n\
19.0580136992706, 48.7184355208188, 65, -1, 0, -1, 1\n\
17.5592067302558, 48.3616064150828, 66, 1, 1, 12, 2\n\
17.1667289342496, 48.1659431503617, 67, 1, 1, 12, 2\n\
18.0393456459588, 48.3145569982147, 68, 1, 1, 12, 2\n\
17.0456355310395, 48.1535468109688, 69, -1, 0, -1, 1\n\
18.7919934009986, 49.1824056669124, 70, 1, 5, 13, 3\n\
18.7352861605213, 49.1836066514882, 71, 1, 5, 14, 3\n\
17.2084667547399, 48.1208644392952, 72, -1, 0, 11, 2\n\
17.1884168040689, 48.1355272794169, 73, -1, 1, 11, 2\n\
18.7684709683779, 49.2041642130102, 74, -1, 0, -1, -1\n\
17.1846907448722, 48.0689307384295, 75, -1, 1, 11, 2\n\
18.0944375141705, 48.3265097313203, 76, 1, 1, 11, 2\n\
17.039515541186, 48.15694263623, 77, -1, 0, -1, 2\n\
21.1677034077035, 48.9820044868356, 78, -1, 0, -1, -1\n\
21.2849568245203, 48.6874358291196, 79, -1, 0, -1, -1\n\
18.7317978482343, 49.2208313675289, 80, -1, 0, -1, 2\n\
17.1092605848406, 48.1102418905338, 81, -1, 0, 11, 2\n\
17.1192405632882, 48.1416635088144, 82, 1, 1, 12, 2\n\
21.2099432685271, 49.0079689386151, 83, 2, 5, 14, 3\n\
18.6963223455319, 49.1826019318607, 84, -1, 0, -1, -1\n\
17.1010529525938, 48.0850947863495, 85, -1, 1, 11, 2\n\
21.2291244115089, 48.7537810852719, 86, 1, 5, 13, 3\n\
17.1504807326246, 48.1773372630238, 87, 1, 4, 13, 3\n\
21.2881736786559, 48.705567511399, 88, 2, 6, 14, 6\n\
21.1820116653743, 48.6299750387021, 89, 1, 5, 14, 3\n\
17.0089082321153, 48.1592982726714, 90, -1, 0, -1, 2\n\
18.0652018802412, 48.2897212659953, 91, -1, 0, -1, 2\n\
19.122376132436, 48.6866089444879, 92, 1, 1, 12, 2\n\
18.6749607852253, 49.2406086522986, 93, 1, 1, 11, 2\n\
18.7305049281708, 49.2295779630874, 94, -1, 0, -1, 1\n\
17.5733126010058, 48.3950632442014, 95, 1, 1, 11, 2\n\
19.0986302816487, 48.8016576320629, 96, -1, 0, 11, 2\n\
17.1689829729612, 48.147447907837, 97, 1, 1, 12, 2\n\
21.1917858016715, 48.6460486498822, 98, 1, 1, 12, 2\n\
17.1611149255593, 48.2063789784255, 99, 1, 5, 13, 3\n\
17.5856587828356, 48.4160591122736, 100, 2, 6, 14, 5\n\
18.099007734618, 48.3152594831971, 101, 1, 1, 13, 3";
	auto database = std::make_shared<Database>("testDatabase13", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter(inputString, "valid_header", true, ',');
	importer.ImportTables(database);

	ASSERT_EQ(true, database->GetTables().find("valid_header") != database->GetTables().end());
	Database::RemoveFromInMemoryDatabaseList("testDatabase13");
}

TEST(CSVDataImportTests, ImportFromString)
{
	const auto inputString = "longitude,latitude,targetId,genderId,ageId,wealthIndexId,hwOsId\n\
17.1911813399913, 48.2262068066084, 1, -1, 1, 11, 2\n\
17.1120308046253, 48.1916689725291, 2, -1, 0, 11, 2\n\
17.137051565592, 48.1963514926608, 3, 1, 1, 13, 3\n\
19.1030194809568, 48.7087577518036, 4, -1, 0, 11, 2\n\
19.0297830535662, 48.7446182349808, 5, 1, 1, 12, 2\n\
17.1642015401874, 48.1739705574233, 6, 2, 6, 14, 3\n\
17.1575513430304, 48.0435042254544, 7, -1, 0, 11, 2\n\
18.1108274310214, 48.3230460790016, 8, 2, 5, 14, 3\n\
21.260598051649, 48.9893752781195, 9, -1, 0, -1, 1\n\
17.154569259203, 48.2056729014943, 10, 1, 0, -1, -1\n\
21.2801768571342, 48.7217686380594, 11, -1, 1, 11, 2\n\
21.2282657634477, 49.0175952916438, 12, 1, 1, 12, 2\n\
18.0920688252035, 48.3226431710195, 13, -1, 0, -1, -1\n\
17.1336134639847, 48.157690914556, 14, -1, 0, -1, 1\n\
21.2157960495558, 48.9834671618524, 15, -1, 0, 11, 2\n\
17.1457386025827, 48.136395108473, 16, -1, 0, 11, 2\n\
21.2359807907819, 49.0314101229962, 17, 1, 1, 12, 2\n\
17.032993940167, 48.2393311844409, 18, 1, 1, 12, 2\n\
18.70530088522, 49.2474893771916, 19, -1, 0, 11, 2\n\
19.1296149605427, 48.7354566926547, 20, 1, 5, 13, 3\n\
21.1980822796886, 48.7790385330901, 21, -1, 0, 11, 2\n\
21.2271508085085, 48.7368486690076, 22, 1, 1, 12, 2\n\
17.6022402830144, 48.378249224587, 23, 2, 6, 14, 3\n\
18.7917811019651, 49.243018912244, 24, 1, 1, 12, 2\n\
21.1734535513608, 48.9779014797044, 25, -1, 1, 11, 2\n\
21.2807242670816, 49.0270912557478, 26, 1, 1, 11, 2\n\
19.101353895609, 48.7546941811633, 27, -1, 0, -1, 2\n\
18.0915388196816, 48.2822852650398, 28, -1, 0, 11, 2\n\
17.2206981686957, 48.195116651325, 29, 1, 1, 12, 2\n\
19.040371984722, 48.7273600658011, 30, -1, 0, 11, 2\n\
17.0322670403572, 48.2406644394767, 31, 2, 6, 14, 3\n\
19.1608426128037, 48.7429181440887, 32, 1, 5, 14, 3\n\
18.1186785568142, 48.3085294040525, 33, 2, 6, 14, 7\n\
18.7229753848342, 49.2486112774919, 34, -1, 1, 11, 2\n\
21.2650211369085, 48.9701319418828, 35, 2, 6, 14, 4\n\
21.2636671412281, 48.9721128681997, 36, 1, 1, 11, 2\n\
18.120813123356, 48.3102668536849, 37, 1, 1, 12, 2\n\
17.1510714605932, 48.1675317621538, 38, -1, 0, 11, 2\n\
21.1723143887861, 48.7689671519407, 39, -1, 0, -1, 2\n\
17.1103912723681, 48.0637726295506, 40, -1, 0, 11, 2\n\
17.1670579704554, 48.0163704172166, 41, -1, 0, 11, 2\n\
17.2699352230583, 48.1076503885326, 42, 2, 5, 14, 3\n\
21.2658830052322, 48.6950746034107, 43, -1, 1, 11, 2\n\
19.1737545783225, 48.7376538723141, 44, 2, 5, 14, 3\n\
17.1483632705747, 48.1537431997214, 45, 1, 1, 12, 2\n\
21.1831327986519, 48.7246185575466, 46, -1, 0, 11, 2\n\
17.0767776101525, 48.1578410845609, 47, -1, 0, -1, 1\n\
18.7390742239502, 49.2656559441761, 48, 1, 1, 12, 2\n\
18.0599425459319, 48.3393543010614, 49, -1, 1, 11, 2\n\
21.2988151632228, 48.727472390728, 50, 1, 1, 12, 2\n\
17.104826746637, 48.1690277255266, 51, 2, 5, 14, 3\n\
17.1606796798867, 48.0431871609626, 52, -1, 0, -1, 2\n\
21.2153329789171, 48.739104195403, 53, -1, 0, -1, -1\n\
18.096698803621, 48.3109541860706, 54, -1, 0, -1, -1\n\
21.2537470990847, 48.9579237058612, 55, 1, 1, 13, 3\n\
19.2288265238046, 48.7434867923389, 56, 1, 5, 14, 3\n\
19.1313071451152, 48.7037611304646, 57, -1, 0, 11, 2\n\
19.116268974379, 48.697737866397, 58, -1, 0, -1, -1\n\
17.2200301181474, 48.1407006709587, 59, 1, 5, 14, 3\n\
17.0595892006296, 48.2381512568764, 60, -1, 0, -1, 1\n\
21.1582436189554, 48.6329263151003, 61, 1, 5, 14, 3\n\
17.1178784431167, 48.0932591531887, 62, 2, 6, 14, 6\n\
17.1606132169184, 48.1111278920224, 63, -1, 0, 11, 2\n\
18.069711576972, 48.3056254548109, 64, 2, 5, 14, 3\n\
19.0580136992706, 48.7184355208188, 65, -1, 0, -1, 1\n\
17.5592067302558, 48.3616064150828, 66, 1, 1, 12, 2\n\
17.1667289342496, 48.1659431503617, 67, 1, 1, 12, 2\n\
18.0393456459588, 48.3145569982147, 68, 1, 1, 12, 2\n\
17.0456355310395, 48.1535468109688, 69, -1, 0, -1, 1\n\
18.7919934009986, 49.1824056669124, 70, 1, 5, 13, 3\n\
18.7352861605213, 49.1836066514882, 71, 1, 5, 14, 3\n\
17.2084667547399, 48.1208644392952, 72, -1, 0, 11, 2\n\
17.1884168040689, 48.1355272794169, 73, -1, 1, 11, 2\n\
18.7684709683779, 49.2041642130102, 74, -1, 0, -1, -1\n\
17.1846907448722, 48.0689307384295, 75, -1, 1, 11, 2\n\
18.0944375141705, 48.3265097313203, 76, 1, 1, 11, 2\n\
17.039515541186, 48.15694263623, 77, -1, 0, -1, 2\n\
21.1677034077035, 48.9820044868356, 78, -1, 0, -1, -1\n\
21.2849568245203, 48.6874358291196, 79, -1, 0, -1, -1\n\
18.7317978482343, 49.2208313675289, 80, -1, 0, -1, 2\n\
17.1092605848406, 48.1102418905338, 81, -1, 0, 11, 2\n\
17.1192405632882, 48.1416635088144, 82, 1, 1, 12, 2\n\
21.2099432685271, 49.0079689386151, 83, 2, 5, 14, 3\n\
18.6963223455319, 49.1826019318607, 84, -1, 0, -1, -1\n\
17.1010529525938, 48.0850947863495, 85, -1, 1, 11, 2\n\
21.2291244115089, 48.7537810852719, 86, 1, 5, 13, 3\n\
17.1504807326246, 48.1773372630238, 87, 1, 4, 13, 3\n\
21.2881736786559, 48.705567511399, 88, 2, 6, 14, 6\n\
21.1820116653743, 48.6299750387021, 89, 1, 5, 14, 3\n\
17.0089082321153, 48.1592982726714, 90, -1, 0, -1, 2\n\
18.0652018802412, 48.2897212659953, 91, -1, 0, -1, 2\n\
19.122376132436, 48.6866089444879, 92, 1, 1, 12, 2\n\
18.6749607852253, 49.2406086522986, 93, 1, 1, 11, 2\n\
18.7305049281708, 49.2295779630874, 94, -1, 0, -1, 1\n\
17.5733126010058, 48.3950632442014, 95, 1, 1, 11, 2\n\
19.0986302816487, 48.8016576320629, 96, -1, 0, 11, 2\n\
17.1689829729612, 48.147447907837, 97, 1, 1, 12, 2\n\
21.1917858016715, 48.6460486498822, 98, 1, 1, 12, 2\n\
17.1611149255593, 48.2063789784255, 99, 1, 5, 13, 3\n\
17.5856587828356, 48.4160591122736, 100, 2, 6, 14, 5\n\
18.099007734618, 48.3152594831971, 101, 1, 1, 13, 3";
	auto database = std::make_shared<Database>("testDatabase14", 1024);
	Database::AddToInMemoryDatabaseList(database);

	CSVDataImporter importer = CSVDataImporter(inputString,"valid_header", true, ',');
	importer.ImportTables(database);

	ASSERT_EQ(101, dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("valid_header")->second.GetColumns().at("targetId").get())->GetBlocksList().front()->GetSize());
	ASSERT_EQ(11, dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("valid_header")->second.GetColumns().at("targetId").get())->GetBlocksList().front()->GetData()[10]);
	ASSERT_EQ(21.2282657634477f, dynamic_cast<ColumnBase<float>*>(database->GetTables().find("valid_header")->second.GetColumns().at("longitude").get())->GetBlocksList().front()->GetData()[11]);
	ASSERT_EQ(-1, dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("valid_header")->second.GetColumns().at("genderId").get())->GetBlocksList().front()->GetData()[12]);
	ASSERT_EQ(3, dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().find("valid_header")->second.GetColumns().at("hwOsId").get())->GetBlocksList().front()->GetData()[100]);
	Database::RemoveFromInMemoryDatabaseList("testDatabase14");
}