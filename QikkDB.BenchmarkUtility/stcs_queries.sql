SELECT Target.targetId, cellId FROM TargetTraffic JOIN Target ON TargetTraffic.targetId = Target.targetId ORDER BY Target.targetId DESC LIMIT 500 OFFSET 30;
9999
SELECT cellId, COUNT(genderId) FROM TargetTraffic JOIN Target ON TargetTraffic.targetId = Target.targetId GROUP BY cellId;
9999
SELECT hwOsId, MIN(TargetTraffic.targetId), MAX(TargetTraffic.eventHour) FROM Target JOIN TargetTraffic ON Target.targetId = TargetTraffic.targetId GROUP BY hwOsId;
9999
SELECT customerId, COUNT(genderId), MAX(ageId) FROM Target JOIN TargetTraffic ON Target.targetId = TargetTraffic.targetId WHERE TargetTraffic.eventHour > 10 GROUP BY customerId;
9999
SELECT dataId, TargetTraffic.eventHour FROM Target JOIN TargetTraffic ON Target.targetId = TargetTraffic.targetId WHERE genderId != -1;
9999