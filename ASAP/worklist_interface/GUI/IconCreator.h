#ifndef __ASAP_GUI_ICONCREATOR__
#define __ASAP_GUI_ICONCREATOR__

#include <mutex>
#include <string>
#include <vector>

#include <QObject>
#include <QStandardItemModel>
#include "ThumbnailCache.h"

namespace ASAP
{
	/// <summary>
	/// Creates thumbnail icons based on WSIs readable by the multiresolution reader.
	/// </summary>
	class IconCreator : public QObject
	{
		Q_OBJECT

		public:
			IconCreator(void);
			~IconCreator(void);
	
			bool insertIcon(const std::pair<int, std::string>& index_location);
			QIcon createIcon(const std::string& filepath, const size_t size);
		
		private:
			QIcon createBlankIcon();
			QIcon createInvalidIcon();
			QIcon m_placeholder_icon;
			QIcon m_invalid_icon;
			ThumbnailCache* m_thumbnail_cache;
			static int m_icon_size;


		signals:
			void requiresStatusBarChange(const QString& message);
			void requiresItemRefresh(int, const QIcon&);
	};
}
#endif // __ASAP_GUI_ICONCREATOR__