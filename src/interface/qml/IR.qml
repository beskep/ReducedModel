import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15

CustomBox {
    ColumnLayout {
        anchors.fill : parent

        RowLayout {
            Layout.fillWidth : true

            TextField {
                Layout.fillWidth : true
                Layout.fillHeight : true
                placeholderText : 'Average [℃]'
            }
            TextField {
                Layout.fillWidth : true
                Layout.fillHeight : true
                placeholderText : 'Minimum [℃]'
            }
            TextField {
                Layout.fillWidth : true
                Layout.fillHeight : true
                placeholderText : 'Maximum [℃]'
            }
        }
    }
}
