#include "TemporaryDirectoryTracker.h"

#include <map>
#include <stdexcept>


namespace ASAP
{
	TemporaryDirectoryTracker::TemporaryDirectoryTracker(const fs::path directory, const TemporaryDirectoryConfiguration configuration) : m_configuration(configuration), m_continue(true), m_directory(directory)
	{
		if (fs::exists(m_directory) && fs::is_regular_file(m_directory))
		{
			throw std::runtime_error("Unable to initialize a file as temporary directory.");
		}
		else
		{
			fs::create_directories(m_directory);
		}

		m_update_thread = std::thread(&TemporaryDirectoryTracker::update, this);
	}

	TemporaryDirectoryTracker::~TemporaryDirectoryTracker(void)
	{
		m_continue = false;
		m_update_thread.join();

		if (m_configuration.clean_on_deconstruct)
		{
			fs::remove_all(m_directory);
		}
	}

	TemporaryDirectoryConfiguration TemporaryDirectoryTracker::getStandardConfiguration(void)
	{
		return { true, true, 0, 5000 };
	}

	fs::path TemporaryDirectoryTracker::getAbsolutePath(void) const
	{
		return fs::absolute(m_directory);
	}

	std::vector<fs::path> TemporaryDirectoryTracker::getFilepaths(void) const
	{
		std::vector<fs::path> filepaths;
		for (auto& entry : fs::directory_iterator(m_directory))
		{
			filepaths.push_back(entry.path());
		}
		return filepaths;
	}

	uint64_t TemporaryDirectoryTracker::getDirectorySizeInMb(void) const
	{
		uint64_t size = 0;
		for (fs::recursive_directory_iterator it(m_directory); it != fs::recursive_directory_iterator(); ++it)
		{
			if (!fs::is_directory(*it))
			{
				size += fs::file_size(*it) / 1e+6;
			}
		}

		return size;
	}

	void TemporaryDirectoryTracker::update(void)
	{
		while (m_continue)
		{
			size_t directory_size = getDirectorySizeInMb();
			if (directory_size > m_configuration.max_size_in_mb)
			{
				std::vector<fs::path> filepaths(getFilepaths());
				std::map<uint64_t, fs::path*> date_sorted_files;
				for (fs::path& p : filepaths)
				{
					date_sorted_files.insert({ static_cast<uint64_t>(fs::last_write_time(p).time_since_epoch().count()), &p });
				}

				for (auto it = date_sorted_files.begin(); it != date_sorted_files.end(); ++it)
				{
					if ((directory_size <= m_configuration.max_size_in_mb) ||
						(it == date_sorted_files.end()-- && m_configuration.allow_overflow))
					{
						break;
					}

					directory_size -= fs::file_size(*it->second) / 1e+6;
					fs::remove(*it->second);
				}
			}
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}
	}
}