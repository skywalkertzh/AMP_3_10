#include "HTTPFileDownload.h"

#include <stdexcept>	

#include "core/stringconversion.h"

namespace ASAP
{
	fs::path httpFileDownload(const web::http::http_response& response, const fs::path& output_directory, std::string output_file, std::function<void(uint8_t)> observer)
	{
		// Fails if the path doesn't point towards a directory.
		if (!fs::is_directory(output_directory))
		{
			// Replace with filesystem error once it has been moved out of experimental
			throw std::runtime_error("Defined directory isn't avaible or not a directory.");
		}

		// Fails if the response wasn't a HTTP 200 message, or lacks the content disposition header.
		web::http::http_headers headers(response.headers());
		auto content_length			= headers.find(L"Content-Length");
		if (response.status_code() == web::http::status_codes::OK && content_length != headers.end())
		{
			// Appends the filename to the output directory.
			fs::path output_file(output_directory / fs::path(output_file));
				
			// Checks if the file has already been downloaded.
			size_t length(std::stoi(content_length->second));
			if (fileIsUnique(output_file, length))
			{
				// Changes filename if the binary size is unique, but the filename isn't.
				fixFilepath(output_file);

				// Fails if the file can't be created and opened.
				concurrency::streams::ostream stream;
				concurrency::streams::fstream::open_ostream(output_file.wstring()).then([&stream](concurrency::streams::ostream open_stream)
				{
					stream = open_stream;
				}).wait();

				if (stream.is_open())
				{
					// Starts monitoring thread.
					bool finished = false;
					std::thread thread(startMonitorThread(finished, length, stream, observer));
					response.body().read_to_end(stream.streambuf()).wait();
					stream.close().wait();
						
					// Joins monitoring thread.
					thread.join();

					if (fileHasCorrectSize(output_file, length))
					{
						return fs::absolute(output_file);
					}
					throw std::runtime_error("Unable to complete download.");
				}

				// Replace with filesystem error once it has been moved out of experimental
				throw std::runtime_error("Unable to create file: " + output_file.string());
			}
			// File has already been downloaded.
			{
				return fs::absolute(output_file);
			}
		}
		throw std::invalid_argument("HTTP Response contains no attachment.");
	}

		bool fileHasCorrectSize(const fs::path& filepath, size_t size)
		{
			return fs::exists(filepath) && fs::file_size(filepath) == size;
		}

		bool fileIsUnique(const fs::path& filepath, size_t size)
		{
			if (fs::exists(filepath) && fs::file_size(filepath) == size)
			{
				return false;
			}
			return true;
		}

		void fixFilepath(fs::path& filepath)
		{
			while (fs::exists(filepath))
			{
				std::string filename = filepath.filename().string();

				size_t version			= 1;
				size_t version_location = filename.find('(');
				if (version_location != std::string::npos)
				{
					size_t value_start	= filename.find_last_of('(') + 1;
					size_t value_end	= filename.find_last_of(')');
					version = std::stoi(filename.substr(value_start, value_end - value_start)) + 1;
				}

				size_t dot_location = filename.find_first_of('.');
				std::string new_filename;
				if (version_location != std::string::npos || dot_location != std::string::npos)
				{
					size_t split_location	= version_location != std::string::npos ? version_location : dot_location;
					new_filename			= filename.substr(0, split_location) + "(" + std::to_string(version) + ")" + filename.substr(filename.find_first_of('.'));
				}
				else
				{
					new_filename = filename + "(" + std::to_string(version) + ")";
				}
				
				filepath.remove_filename() /= new_filename;
			}
		}

		std::thread startMonitorThread(const bool& stop, const size_t length, concurrency::streams::ostream& stream, std::function<void(uint8_t)>& observer)
		{
			return std::thread([&stop, length, &stream, observer](void)
				{
					// If there is no observer, we don't need to report progress.
					if (observer)
					{
						// Keeps checking progress until the stream is closed, or the download has completed.
						try
						{
							size_t percentile = (length / 100);
							size_t progress = stream.tell();
							while (!stop && progress < length)
							{
								observer(static_cast<float>(progress / percentile));
								std::this_thread::sleep_for(std::chrono::seconds(1));
								progress = stream.tell();
							}
						}
						catch (...)
						{
							// No need to handle this. If triggered, the stream has closed, and we no longer need to provide progress.
						}
					}
				});
		}
}