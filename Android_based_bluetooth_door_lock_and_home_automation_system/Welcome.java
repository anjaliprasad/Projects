package project.door.lock.pack;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.LinearLayout;

public class Welcome extends Activity implements OnClickListener{

	LinearLayout welcome;
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		// TODO Auto-generated method stub
		super.onCreate(savedInstanceState);
		setContentView(R.layout.welcome);
		
		welcome=(LinearLayout)findViewById(R.id.welcome_page);
		welcome.setOnClickListener(this);
	}
	public void onClick(View v) {
		// TODO Auto-generated method stub
		
		if(v.equals(welcome))
		{
			Intent it=new Intent(getBaseContext(),MainActivity.class);
			startActivity(it);
			
		}
	}

}
