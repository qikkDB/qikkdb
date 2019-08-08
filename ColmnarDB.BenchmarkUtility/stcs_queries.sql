SELECT Target.targetId, cellId FROM TargetTraffic JOIN Target ON TargetTraffic.targetId = Target.targetId ORDER BY Target.targetId DESC LIMIT 500 OFFSET 30;
SELECT cellId, COUNT(genderId) FROM TargetTraffic JOIN Target ON TargetTraffic.targetId = Target.targetId GROUP BY cellId;
SELECT hwOsId, MIN(TargetTraffic.cellId), MAX(TargetTraffic.eventHour) FROM Target JOIN TargetTraffic ON Target.targetId = TargetTraffic.targetId GROUP BY hwOsId;
SELECT customerId, COUNT(genderId), MAX(ageId) FROM Target JOIN TargetTraffic ON Target.targetId = TargetTraffic.targetId WHERE TargetTraffic.eventHour > 10 GROUP BY customerId;
SELECT dataId, TargetTraffic.eventHour FROM Target JOIN TargetTraffic ON Target.targetId = TargetTraffic.targetId WHERE genderId != -1;