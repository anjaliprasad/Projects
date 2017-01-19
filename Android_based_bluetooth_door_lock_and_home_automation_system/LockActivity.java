package project.door.lock.pack;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.ImageView;

public class LockActivity extends Activity implements OnClickListener
{
	ImageView btn_lock,btn_unlock;
	public void onCreate(Bundle b)
	{
		super.onCreate(b);
		setContentView(R.layout.lock);
		btn_lock=(ImageView)findViewById(R.id.lock_btn_lock);
		btn_lock.setOnClickListener(this);
		
		btn_unlock=(ImageView)findViewById(R.id.lock_btn_unlock);
		btn_unlock.setOnClickListener(this);
	}
	public void onClick(View v) {
		// TODO Auto-generated method stub
		if(v.equals(btn_lock))
		{
			MainActivity.getBluetoothChatObject().write("c".getBytes());
		}
		else if(v.equals(btn_unlock))
		{
			MainActivity.getBluetoothChatObject().write("o".getBytes());
		}
		
	}

}
