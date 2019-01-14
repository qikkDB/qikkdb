#pragma once
#include <array>
#include <vector>
#include "ITCPWorker.h"

class ClientPoolWorker final : ITCPWorker
{
private:
	bool quit_;
public:
	/// <summary>
	/// Create new instance of ClientPoolWorker object
	/// </summary>
	/// <param name="activeWorkers">Instance of object responsible for handling messages</param>
	/// <param name="handler">Instance of object responsible for handling messages</param>
	/// <param name="socket">Client that will be handled by this instance</param>
	/// <param name="requestTimeout">Timeout for TCP read and write in ms</param>
	ClientPoolWorker(std::set<std::shared_ptr<ITCPWorker>>& activeWorkers, std::unique_ptr<IClientHandler>&& clientHandler, boost::asio::ip::tcp::socket socket, int requestTimeout);

	// Inherited via ITCPWorker
	virtual void HandleClient() override;

	// Inherited via ITCPWorker
	virtual void Abort() override;

	ClientPoolWorker(const ClientPoolWorker&) = delete;
	ClientPoolWorker& operator=(const ClientPoolWorker&) = delete;

};

